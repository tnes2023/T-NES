# https://github.com/as791/ZOO_Attack_PyTorch/blob/main/zoo_l2_attack_black.py
import numpy as np
import torch
import random
from numba import jit
import sys
from data.LibriSpeech import TextTransform
from utils import char_error, deepspeech_decoder, td,GreedyDecoder
import pickle
from .td_sim import get_tdsim_indice, sim_loss, td_sim_2
# np.random.seed(0)
# torch.manual_seed(0)
import torch.nn.functional as F



@jit(nopython=True)
def coordinate_ADAM(ori_ctc, losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size,beta1, beta2, proj, random_noise):
      # esitmate gradient by finite difference
      # for i in range(batch_size):
      #     grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002

    # get first order gradient by finite difference
      for i in range(batch_size):
          grad[i] = (losses[0] - losses[1]) / 0.0002

      # ADAM update
      mt = mt_arr[indice]
      mt = beta1 * mt + (1 - beta1) * grad
      mt_arr[indice] = mt
      vt = vt_arr[indice]
      vt = beta2 * vt + (1 - beta2) * (grad * grad)
      vt_arr[indice] = vt
      epoch = adam_epoch[indice]
      corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
      m = real_modifier.reshape(-1)
      old_val = m[indice]
      old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
      # set it back to [-0.5, +0.5] region
      if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])

      if np.min(old_val) == 0 and np.max(old_val) == 0:
          # print(1)
          m[indice] = random_noise
      else:
          m[indice] = old_val
      adam_epoch[indice] = epoch + 1


@jit(nopython=True)
def coordinate_Newton(ori_ctc, losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down,
                      step_size, beta1, beta2, proj, random_noise):
    cur_loss = losses[0] # ori image loss

    # for i in range(batch_size):
    #     grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.002
    #     hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (0.001 * 0.001)

    for i in range(batch_size):
        grad[i] = (losses[0] - losses[1]) / 0.0002 # first order
        hess[i] = (losses[0] - 2*cur_loss + losses[1])/ (0.001 * 0.001) # second order

    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= step_size * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])


    if np.min(old_val) == 0 and np.max(old_val) == 0:
        print(1)
        m[indice] = random_noise
    else:
        m[indice] = old_val


def loss_run(input, spectrograms_clip, target, sepc_len, model, modifier,ctc_loss, model_name='ds2'):
    # print(spectrograms_clip)
    modifier_ts = torch.clip(modifier, spectrograms_clip.min().item(), spectrograms_clip.max().item()) * 8

    pert_out = input + modifier_ts
    l2penalty = 1

    # compute  CTCLoss for each models
    if model_name == 'wave2letter':
        y_hat = model(pert_out)
        output = y_hat.permute(2, 0, 1)  # TxNxH
        output = output.log_softmax(-1)
        float_out = output.float()  # ensure float32 for loss
        target = target.repeat(modifier.shape[0],1)
        target_len = torch.from_numpy(np.array([target.size(-1)]*modifier.shape[0])).to("cuda", dtype=torch.int32)

        loss2 = ctc_loss(float_out, target,
                            torch.IntTensor([float_out.shape[0]] * len(target_len)).cuda(),
                            target_len)
        outsizes = 0
    elif model_name == 'ds2':
        output,outsizes = model(pert_out,torch.IntTensor(sepc_len*pert_out.shape[0]).cuda()) # batch_size, audio size
        out = output.transpose(0, 1)
        out = out.log_softmax(-1)

        output = out.transpose(1, 0)
        target = target.repeat(modifier.shape[0],1)
        target_len = torch.from_numpy(np.array([target.size(-1)]*modifier.shape[0])).to("cuda", dtype=torch.int32)
        loss2 = ctc_loss(out, target, outsizes, target_len)

    elif model_name == 'ds2v1':
        out = model(pert_out)
        output = F.log_softmax(out, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        # input_lengths = [length // 2 for length in sepc_len]
        target = target.repeat(modifier.shape[0],1)

        target_len = torch.from_numpy(np.array([target.size(-1)]*modifier.shape[0])).to("cuda", dtype=torch.int32)

        loss2 = ctc_loss(output,
                            target,
                            torch.IntTensor(sepc_len * len(target_len)).cuda(),
                            target_len)

        outsizes = 0

    # loss1 = sim_loss(pert_out,h=3)
    # distance between perturbed audios and orginal audios
    if model_name == 'wave2letter':
        loss1 = torch.sum(torch.square(pert_out - input),dim=(1,2))
    else:
        loss1 = torch.sum(torch.square(pert_out - input),dim=(1,2,3))

    # optimization function for CW attack
    # maximize the ctcloss between perturbed audio and original inputs
    if not np.isinf(l2penalty):
        loss = loss1 - l2penalty*loss2 # l2penalty = const
    else:
        loss = -loss2
    return outsizes,loss.detach().cpu().numpy(), loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()


def l2_attack(input, ori_input, target, sepc_len, model, loss_func, targeted, ori_cer, solver, ori_ctc, B, evaluation_decoder, f, txt_writer, idx,
              batch_size=100, cw_iter=1000, step_size=0.1, adam_beta1=0.9, adam_beta2=0.999, use_TD=0, model_name='ds2'):
    """https://github.com/huanzhang12/ZOO-Attack/blob/master/l2_attack_black.py"""
    generated_audios = {idx:[]}
    # f = open(str(solver)+str(batch_size)+str(B)+"pkl","wb")
    var_len = input.reshape(-1).size()[0]
    modifier_up = np.zeros(var_len, dtype=np.float32)
    modifier_down = np.zeros(var_len, dtype=np.float32)
    real_modifier = torch.zeros(input.size(), dtype=torch.float32).cuda()
    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32) # coodinate amount
    hess = np.zeros(batch_size, dtype=np.float32)
    h = 3  # set window len
    total_q= 0
    var_list = np.array(range(0, var_len), dtype=np.int32)
    cer_certierion = 0.1
    pert_input = input.detach().cpu().numpy()
    input_min = torch.min(ori_input)
    input_max = torch.max(ori_input)


    spectrograms_clip = ori_input
    cer_list = []
    # prev_g = np.zeros(shape=input.size())
    prev_indice = []
    # momentum = 0.9
    change_list = []
    for iter in range(cw_iter): # cw attack iteration
        # get indices
        if use_TD == 0:
            indices = np.random.choice(input.size()[-1] * input.size()[-2], size=batch_size, replace=False)
        else:
            if total_q <= 3:  # initalize
                var_list = np.array(range(0, input.reshape(-1).size()[0]), dtype=np.int32)
                indices = var_list[np.random.choice(var_list.size, int(batch_size), replace=False)]  # selected coord index
            else:  # after the first iteration
                _, _, indice = td(ori_input, pert_input, input.size()[-1], h, batch_size, prev_indice)
                prev_indice += indice.tolist()
                indices = np.array(indice)
        change_list += indices.tolist()
        # print(len(set(change_list)))
        random_noise = np.random.normal(size=indices.shape)
            # if len(cer_list) < 10:
            #     h = 3
            #     if total_q <= 3:  # initalize
            #         var_list = np.array(range(0, input.reshape(-1).size()[0]), dtype=np.int32)
            #         indices = var_list[np.random.choice(var_list.size, int(batch_size), replace=False)]  # selected coord index
            #     else:  # after the first iteration
            #         _, _, indice = td(ori_input, pert_images, input.size()[-1], h, batch_size)
            #         indices = np.array(indice)
            # elif cer_list[-1] <= cer_list[-10]:
            #         indices = np.random.choice(input.size()[-1] * input.size()[-2], size=batch_size, replace=False)
            # else:
            # _, _, indice = td(ori_input, pert_images, input.size()[-1], h, batch_size)
            # indices = np.array(indice)
            # indices, change_list = get_tdsim_indice(input, batch_size=batch_size, h=10, change_list=change_list)
            # indices = td_sim_2(input, noise_samples=batch_size, h=3).detach().cpu().numpy()
            # print(indices[:10])

        # var = np.repeat(real_modifier.detach().cpu().numpy(), 2, axis=0)
        # evaluation points for fintie difference
        var = np.repeat(np.zeros(shape=input.size()), 2, axis=0)
        # for i in range(batch_size):
        #     var[i+1].reshape(-1)[indice[i]] += 0.0001
        #     var[i+2].reshape(-1)[indice[i]] -= 0.0001
        var[0].reshape(-1)[indices] += 0.0001
        var[1].reshape(-1)[indices] -= 0.0001

        var = torch.tensor(var).to("cuda",dtype=torch.float32)
        var = var.reshape((-1,) + input.size()[1:])

        # get perturbated audio loss
        outsize, losses, l2s, losses2, scores, pert_images = loss_run(input, spectrograms_clip,
                                                                      target, sepc_len, model, var,loss_func, model_name=model_name)
        # losses: ctc+distance; l2s: l2 distance; losses2: ctcloss; scores: output_latest_ds2v1 matrix
        # print(losses)

        real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()

        # print(np.max(real_modifier_numpy), np.min(real_modifier_numpy))
        # update solvers' hyperparameters based on the loss
        if solver == "adam":
            # print("adam epoch", iter, np.mean(real_modifier_numpy))
            coordinate_ADAM(ori_ctc, losses2, indices, grad, hess, batch_size, mt, vt, real_modifier_numpy, adam_epoch,
                            modifier_up, modifier_down, step_size, adam_beta1, adam_beta2,proj=False, random_noise=random_noise)

        if solver == "newton":
            coordinate_Newton(ori_ctc, losses2, indices, grad, hess, batch_size, mt, vt, real_modifier_numpy, adam_epoch,
                              modifier_up, modifier_down, step_size, adam_beta1, adam_beta2,proj=False, random_noise=random_noise)
        # real_modifier_numpy = np.clip(real_modifier_numpy, -B, B)

        # common_indice = np.intersect1d(indices, prev_indice)
        # common_indice = common_indice.astype('int32')
        # if model_name == 'wave2letter':
        #     tmp_g = np.zeros(shape=input.size()[1] * input.size()[2])
        # else:
        #     tmp_g = np.zeros(shape=input.size()[2] * input.size()[3])
        # tmp_g[common_indice] = prev_g.reshape(-1)[common_indice]
        # tmp_g = tmp_g.reshape(input.size())
        # real_modifier_numpy = momentum * tmp_g + (1.0 - momentum) * real_modifier_nump
        # prev_g = real_modifier_numpy
        # prev_indice = indices

        # print(np.min(real_modifier_numpy), np.max(real_modifier_numpy))
        real_modifier = torch.from_numpy(real_modifier_numpy).cuda()
        real_modifier = real_modifier.float()

        # check attack results
        output_size, loss, l2, loss2, model_out, pert_input = loss_run(input, spectrograms_clip, target, sepc_len, model, real_modifier, loss_func, model_name=model_name)
        if iter % 50 == 0: # update perturbations every 50 iterations
            input = torch.from_numpy(pert_input).cuda()
            real_modifier = torch.zeros(input.size(), dtype=torch.float32).cuda()

        if model_name == 'wave2letter':
            out = torch.from_numpy(model_out).cuda()
            out_decode = out.clone()
            out_decode = out_decode.transpose(0, 1)
            decoded_output, _ = evaluation_decoder.decode(out_decode, [out_decode.shape[1]])
        elif model_name == 'ds2':
            decoded_output, _ = evaluation_decoder.decode(torch.from_numpy(model_out).cuda(), output_size)
        elif model_name == 'ds2v1':
            out = torch.from_numpy(model_out).cuda()
            out = out.transpose(0, 1)
            decoded_output, _ = evaluation_decoder.decode(out.cuda(), sepc_len)

        adv_cer = evaluation_decoder.cer(decoded_output[0][0], targeted) / (len(decoded_output[0][0])+0.0001)
        ori_wer = evaluation_decoder.wer(decoded_output[0][0], targeted) / (len(decoded_output[0][0].split(' '))+0.0001)

        increased_cer = adv_cer - ori_cer
        total_q += 1
        print("query times:", total_q, "ori CER:", ori_cer, "adv CER:", adv_cer, "increase by:", increased_cer, file=txt_writer)
        print("query times:", total_q, "ori CER:", ori_cer, "adv CER:", adv_cer, "increase by:", increased_cer)
        cer_list.append(increased_cer)
        if increased_cer > cer_certierion:
            generated_audios[idx].append(pert_input) # save increased CER generated AEs
            # pickle.dump(pert_input, f)
            cer_certierion += 0.1
            # print("query = {}, ori CER = {:.5f}, adv CER = {:.5f}, increase by = {:.5f}".
            #       format(total_q, ori_cer, adv_cer, increased_cer), file=txt_writer)
            print("query = {}, ori CER = {:.5f}, adv CER = {:.5f}, increase by = {:.5f}".
                  format(total_q, ori_cer, adv_cer, increased_cer), file=txt_writer)
            if increased_cer >= 0.6:
                break
    pickle.dump(generated_audios, f)
        # if losses2[0] == 0.0 and last_loss2 != 0.0 and stage == 0:
        #     if reset_adam_after_found:
        #         mt.fill(0)
        #         vt.fill(0)
        #         adam_epoch.fill(1)
        #     stage = 1
        # last_loss2 = losses2[0]

        # if abort_early and (iter + 1) % early_stop_iters == 0:
        #     if losses[0] > prev * .9999:
        #         print("Early stopping because there is no improvement")
        #         break
        #     prev = losses[0]
        #
        # if l2s[0] < bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(), -1)):
        #     bestl2 = l2s[0]
        #     bestscore = np.argmax(scores[0])

        # if l2s[0] < out_bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(), -1)):
        #     if out_bestl2 == 1e10:
        #         print(
        #             "[STATS][L3](First valid attack found!) iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(
        #                 iter + 1, losses[0], l2s[0], losses2[0]))
        #         sys.stdout.flush()
        #     out_bestl2 = l2s[0]
        #     out_bestscore = np.argmax(scores[0]) #
        #     out_best_attack = pert_images[0]
        #     out_best_const = const

    # if compare(bestscore, np.argmax(target.cpu().numpy(), -1)) and bestscore != -1:
    #     print('old constant: ', const)
    #     upper_bound = min(upper_bound, const)
    #     if upper_bound < 1e9:
    #         const = (lower_bound + upper_bound) / 2
    #     print('new constant: ', const)
    # else:
    #     print('old constant: ', const)
    #     lower_bound = max(lower_bound, const)
    #     if upper_bound < 1e9:
    #         const = (lower_bound + upper_bound) / 2
    #     else:
    #         const *= 10
    #     print('new constant: ', const)



