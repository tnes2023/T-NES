# https://github.com/rtaori/Black-Box-Audio/blob/master/run_audio_attack.py
import pickle
import torch.nn.functional as F
import numpy as np
import torch
from data.LibriSpeech import TextTransform
from utils import deepspeech_decoder, char_error, GreedyDecoder
from attack.fd_nes import fd
import scipy.io.wavfile as wav
import os
import sys
import random
import Levenshtein
from scipy.signal import butter, lfilter
from .td_sim import td_sim_2


def get_loss_batch(model, loss_f, spec, sepc_len, label, label_length, model_name='ds2'):
    spec = torch.from_numpy(spec).to("cuda", dtype=torch.float32)
    if model_name == 'wave2letter':
        spec = spec.squeeze(1)
        y_hat = model(spec)
        out = y_hat.permute(2, 0, 1)  # TxNxH
        out = out.log_softmax(-1)
        float_out = out.float()  # ensure float32 for loss
        l = loss_f(float_out, label,
                            torch.IntTensor([float_out.shape[0]] * len(sepc_len)).cuda(),
                            label_length)
    elif model_name == 'ds2':
        y_hat, output_sizes = model(spec, sepc_len)
        out = y_hat.transpose(0, 1)
        out = out.log_softmax(-1)
        l = loss_f(out, label, output_sizes, label_length)
    elif model_name == 'ds2v1':
        y_hat = model(spec)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        l = loss_f(output,
                   label,
                   sepc_len,
                   label_length)

    return y_hat, l


#
# def highpass_filter(data, cutoff=7000, fs=16000, order=10):
#     b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
#     return lfilter(b, a, data)


def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max())
    elite_pop_probs = scores_logits / scores_logits.sum()
    cand1 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5
    mask = mask[..., np.newaxis, np.newaxis]
    next_pop = mask * cand1 + (1 - mask) * cand2
    return next_pop


def mutate_pop(pop, mutation_p, noise_stdev, elite_pop, spectrograms):
    noise = np.random.randn(*pop.shape) * noise_stdev
    noise = np.clip(noise, -spectrograms.cpu().detach().numpy() * 0.5, spectrograms.cpu().detach().numpy() * 0.5)
    # noise = highpass_filter(noise)
    # mask = np.random.rand(pop.shape[0], elite_pop.shape[1]) < mutation_p
    mask = np.random.rand(elite_pop.shape[-2], elite_pop.shape[-1]) < mutation_p
    mask = mask[np.newaxis, np.newaxis, ...]
    new_pop = pop + noise * mask
    return new_pop


def genetic(model, loss_f, spec, spec_len, label, label_length, target_s,
            ori_cer, ori_ctc, max_iter, batch_size, f, txt_writer, i, B, use_TD=0, model_name='ds2'):
    generated_audios = {i: []}
    spectrograms_clip = spec * B

    itr = 1
    max_fitness_score = float('-inf')
    elite_size = 6
    pop_size = 3
    num_points_estimate = 8
    pop = np.tile(spec.cpu().detach().numpy(), (pop_size, 1, 1, 1))
    dist = float("inf")
    mutation_p = 0.005
    noise_stdev = 4000
    delta_for_gradient = 100
    delta_for_perturbation = 1e3
    mu = 0.5
    alpha = 0.001
    prev_loss = None
    total_query = 0
    modified = np.zeros(shape=spec.size())
    libris_transform = TextTransform()
    labels = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
              "t", "u", "v", "w", "x", "y", "z", " "]
    if model_name == 'wave2letter':
        evaluation_decoder = GreedyDecoder(labels, blank_index=28)
    else:
        evaluation_decoder = GreedyDecoder(labels, blank_index=0)

    cer_criterion = 0.1

    while total_query < max_iter:
        input_len_batch = torch.from_numpy(np.array(spec_len * pop_size)).to("cuda", dtype=torch.int32)
        label_batch = label.repeat([pop_size, 1])
        label_len_batch = torch.from_numpy(np.array([label.shape[1]] * pop_size)).to("cuda", dtype=torch.int32)

        # losses of pop
        _, pop_scores = get_loss_batch(model, loss_f, pop, input_len_batch, label_batch, label_len_batch, model_name=model_name)
        pop_scores = pop_scores.cpu().detach().numpy()
        elite_ind = np.argsort(pop_scores)[-elite_size:]
        elite_pop, elite_ctc = pop[elite_ind], pop_scores[elite_ind]
        best_pop = torch.from_numpy(np.expand_dims(elite_pop[0, ...], axis=0)).to("cuda", dtype=torch.float32)

        # best_pop = torch.clip(best_pop, -spectrograms_clip.item(), spectrograms_clip.item())
        # CER of pop
        if model_name == 'wave2letter':
            best_pop = best_pop.squeeze(1)
            out = model(best_pop)
            out_decode = out.clone()
            out_decode = out_decode.transpose(1, 2)
            decoded_output, _ = evaluation_decoder.decode(out_decode, [out_decode.shape[1]])
        elif model_name == 'ds2':
            output, output_sizes = model(best_pop,
                                         torch.IntTensor(spec_len).cuda())  # TODO ensure large->small
            decoded_output, _ = evaluation_decoder.decode(output, output_sizes)
        elif model_name == 'ds2v1':
            output = model(best_pop)
            decoded_output, _ = evaluation_decoder.decode(output, spec_len)


        tar_str = evaluation_decoder.convert_to_strings(label)
        adv_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / len(decoded_output[0][0])
        increase_cer = adv_cer - ori_cer

        print("total query:", total_query, "adv CER:", adv_cer, "increase CER:", increase_cer, file=txt_writer)

        if increase_cer > cer_criterion:
            cer_criterion += 0.1
            print("*************over {}*******************".format(cer_criterion))
            print("query times:", total_query, "ori CER:", ori_cer, "adv CER:", adv_cer, "increase by:", increase_cer, file=txt_writer)
            print("****************************************")

        if prev_loss is not None and prev_loss != elite_ctc[0]:
            mutation_p = mu * mutation_p + alpha / np.abs(prev_loss - elite_ctc[0])

        if increase_cer < 0.02:
            next_pop = get_new_pop(elite_pop, elite_ctc, pop_size)
            pop = mutate_pop(next_pop, mutation_p, noise_stdev, elite_pop, spec)
            prev_loss = elite_ctc[0]

            total_query += pop_size
    #
        else:  # closed to end then use FD
            break
    #
    adv_spec = best_pop
    cer_list = []
    prev_g = np.zeros(shape=spec.size())
    prev_indice = []
    momentum = 0.9
    while total_query < max_iter:
        total_query += 1
        if use_TD == 0:
            g_hat, indices, prev_indice = fd(model, spec, adv_spec, loss_f, label, spec_len,
                       batch_size, ori_ctc, max_iter - total_query, use_TD=0, model_name=model_name)
        else:
            g_hat, indices, prev_indice = fd(model, spec, adv_spec, loss_f, label, spec_len,
                                                 batch_size, ori_ctc, max_iter - total_query, 1, model_name=model_name,
                                                 prev_indice=prev_indice)




        # g_hat = np.clip(g_hat, -spectrograms_clip.item(), spectrograms_clip.item())
        adv_spec = adv_spec + torch.from_numpy(g_hat).to('cuda', dtype=torch.float)
        # adv_spec = torch.clip(adv_spec, input_min, input_max)
        if model_name == 'wave2letter':
            out = model(adv_spec)
            out_decode = out.clone()
            out_decode = out_decode.transpose(1, 2)
            decoded_output, _ = evaluation_decoder.decode(out_decode, [out_decode.shape[1]])
        elif model_name == 'ds2':
            out, output_sizes = model(adv_spec, torch.IntTensor(spec_len).cuda())
            decoded_output, _ = evaluation_decoder.decode(out, output_sizes)
        elif model_name == 'ds2v1':
            out = model(adv_spec)
            decoded_output, _ = evaluation_decoder.decode(out, spec_len)

        tar_str = evaluation_decoder.convert_to_strings(label)
        adv_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0])+0.0001)
        adv_wer = evaluation_decoder.wer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0].split())+0.0001)
        increase_cer = adv_cer - ori_cer
        cer_list.append(increase_cer)
        if increase_cer > cer_criterion and cer_criterion < 1:

            print("query = {}, ori CER = {:.5f}, adv CER = {:.5f}, increase by = {:.5f}, adv_wer = {:.5f}".
                  format(total_query, ori_cer, adv_cer, increase_cer, adv_wer), file=txt_writer)
            cer_criterion += 0.1
            generated_audios[i].append(adv_spec.cpu().detach().numpy())
            if increase_cer >= 0.6:
                break

        print("query times:", total_query, "ori CER:", ori_cer, "adv CER:", adv_cer, "increase by:", increase_cer,
              "adv_wer:", adv_wer, file=txt_writer)

    pickle.dump(generated_audios, f)

