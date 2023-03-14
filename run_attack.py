from attack.HSJA import hop_skip_jump_attack
from data.LibriSpeech import librispeech_loader, ted_dataloader
from nets.DeepSpeech2_v2 import DeepSpeech
from nets.DeepSpeech2_v1 import DeepSpeech as ds2v1
from nets.Wav2LetterPlus import Wav2LetterPlus
from attack.fd_nes import fd,nes
from attack.zoo_based import l2_attack
from attack.genetic import genetic
from utils import GreedyDecoder
import os
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    if not os.path.exists('output_new/'):
        os.mkdir('output_new')

    #======== load deepspeech2 (LSTM) =======
    if args.model == 'ds2':
        labels = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                  "s", "t", "u", "v", "w", "x", "y", "z", " "]
        model = DeepSpeech(labels=labels,
                           rnn_type=nn.LSTM,  # nn.LSTM, nn.GRU
                           # audio_conf=audio_conf,
                           bidirectional=True)
        if args.dataset == "ls": # ckpt for ls
            ckpt = torch.load('ckpt/deepspeech2_ls.pth')
        if args.dataset == "ted": # ckpt for ted
            ckpt = torch.load('ckpt/ds2_ted.pth')

    # ======== load deepspeech2 (GRU) =======
    elif args.model == 'ds2v1':
        labels = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                  "s", "t", "u", "v", "w", "x", "y", "z", " "]
        model = ds2v1()
        ckpt = torch.load('ckpt/ds2v1_librispeech.pt')

    # ======== load Wav2letter =======
    elif args.model == 'wave2letter':
        labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                  "t", "u", "v", "w", "x", "y", "z", "'", "|"]
        model = Wav2LetterPlus(29)
        ckpt = torch.load('ckpt/wav2letter_plus.pt')

    model.load_state_dict(ckpt)
    model = model.cuda()
    # model = nn.DataParallel(model).cuda()
    ctc_loss = nn.CTCLoss(blank=0, reduction="none")
    # sim_loss = nn.CosineSimilarity()
    model.eval()

    # ======== dataset loader =======
    if args.dataset == "ls":
        data_path = args.data_path
        if args.model == 'wave2letter':
            train_loader, valid_loader, test_loader = librispeech_loader(data_path, batch_size=1, mel_num=128,
                                                                         process_type=2)
        elif args.model == 'ds2':
            train_loader, valid_loader, test_loader = librispeech_loader(data_path,batch_size=1, mel_num=128)
        elif args.model == 'ds2v1':
            train_loader, valid_loader, test_loader = librispeech_loader(data_path,batch_size=1, mel_num=128, model_name='ds2v1')

    if args.dataset =="ted":

        data_path = args.data_patj
        train_loader, valid_loader, test_loader = ted_dataloader(data_path, batch_size=1, mel_num=128)

    #========  hyper-parameters for all methods ========
    samples_per_draw = args.samples_per_draw
    max_query = args.max_query
    grad_est = args.method
    B = args.B

    # NES
    sigma = 0.0001
    momentum = 0.9

    # ZOO
    solver = args.solver
    step_size = 0.01
    adam_beta1 = 0.9
    adam_beta2 = 0.999

    if args.model == 'wave2letter': # decoder for Wav2letter
        evaluation_decoder = GreedyDecoder(labels, blank_index=28)
    else: # decoder for other models
        evaluation_decoder = GreedyDecoder(labels, blank_index=0)

    f = open("output_new/{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(args.dataset, args.model, args.samples_per_draw,
                                                  args.method, args.solver, args.max_query, B, args.use_TD), 'wb')
    txt_writer = open("output_new/{}_{}_{}_{}_{}_{}_{}_{}.txt".format(args.dataset, args.model, args.samples_per_draw,
                                                  args.method, args.solver, args.max_query, B, args.use_TD), 'w')
    change_list = []
    attack_counts = 0
    for idx, i in enumerate(test_loader):
        history_indices = []
        cer_list = []
        generated_audios = {idx:[]}
        cer_criterion = 0.1
        # if idx >= 0 and idx < 20:
        if attack_counts < 20:
            print("==============================", file=txt_writer)
            print("ID:", idx + 1, file=txt_writer)

            spectrograms, labels, input_lengths, label_lengths = i
            spectrograms, labels = spectrograms.to("cuda", dtype=torch.float), labels.to("cuda", dtype=torch.float)


            spectrograms_clip = spectrograms * B # set constraints for esitmated gradients
            spectrograms_clip = spectrograms_clip.detach().cpu().numpy()
            adv_spec = spectrograms.detach()

            # ======== predicted sentence by DeepSpeech(character string) ========
            if args.model == 'wave2letter':
                out = model(spectrograms)
                out_decode = out.clone()
                out_decode = out_decode.transpose(1, 2)
                out = out.permute(2, 0, 1)  # TxNxH
                out = out.log_softmax(-1)
                float_out = out.float()  # ensure float32 for loss
                ori_loss = ctc_loss(float_out, labels,
                                 torch.IntTensor([float_out.shape[0]] * len(input_lengths)).cuda(),
                                 torch.IntTensor(label_lengths).cuda())

                decoded_output, _ = evaluation_decoder.decode(out_decode, [out_decode.shape[1]])


            elif args.model == 'ds2':
                out, output_sizes = model(spectrograms, torch.IntTensor(input_lengths).cuda())
                out_ctc = out.transpose(0, 1)
                out_ctc = out_ctc.log_softmax(-1)
                ori_loss = ctc_loss(out_ctc, labels, output_sizes, torch.IntTensor(label_lengths).cuda())

                decoded_output, _ = evaluation_decoder.decode(out, output_sizes)
            elif args.model == 'ds2v1':
                out = model(spectrograms)
                output = F.log_softmax(out, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)
                input_lengths = [length // 2 for length in input_lengths]
                ori_loss = ctc_loss(output,
                                    labels,
                                    torch.IntTensor(input_lengths).cuda(),
                                    torch.IntTensor(label_lengths).cuda())

                decoded_output, _ = evaluation_decoder.decode(output.transpose(0, 1), torch.IntTensor(input_lengths).cuda())
            tar_str = evaluation_decoder.convert_to_strings(labels)
            if tar_str[0][0] == "ignoretimesegmentinscoring ": # pass audios in TED
                print(tar_str)
                continue

            # compute original cer and wer
            target_sentence = tar_str[0][0]
            ori_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0]) + 0.0001)
            ori_wer = evaluation_decoder.wer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0].split(' ')) + 0.0001)
            print("Decoded Output:", decoded_output)
            print("Target Output: ", tar_str)
            print("CER:", ori_cer, "WER:", ori_wer)

            # set total query as 0 at the beginning
            total_q = 0
            prev_g = np.zeros(shape=spectrograms.size())
            prev_indice = np.array([])

            # compute gradient within query budget
            while total_q<max_query:
                if grad_est == "hsja":
                    g_hat, history_indices = hop_skip_jump_attack(model, args.model, adv_spec,
                                                    torch.IntTensor(input_lengths).cuda(),
                                                    np.inf, spectrograms, history_indices,
                                                    num_iterations=1, batch_size=1,
                                                    clip_min=torch.min(spectrograms) * 2,
                                                    clip_max=torch.max(spectrograms) * 2,
                                                    sample_per_draw=args.samples_per_draw,
                                                    use_TD=args.use_TD,
                                                    total_q=total_q)
                    total_q += 1
                elif grad_est == "fd":
                    if args.use_TD == 0:
                        g_hat, indices, history_indices = fd(model, spectrograms, adv_spec, ctc_loss, labels, input_lengths,
                                   samples_per_draw, ori_loss.item(),total_q,0, model_name=args.model, prev_indice=[])
                    else:
                        g_hat, indices, history_indices = fd(model, spectrograms, adv_spec, ctc_loss, labels, input_lengths,
                                       samples_per_draw, ori_loss.item(), total_q, 1, model_name=args.model, prev_indice=history_indices)

                    total_q += 1

                elif grad_est == "nes":
                    total_q += 1
                    if args.use_TD == 0:
                        _, g_hat, indices, history_indices = nes(model, spectrograms, adv_spec, ctc_loss, labels,
                                                input_lengths, samples_per_draw, sigma, total_q, 0, model_name=args.model, prev_indice=history_indices)
                    else:
                        _, g_hat, indices, history_indices = nes(model, spectrograms, adv_spec, ctc_loss, labels,
                                                    input_lengths, samples_per_draw, sigma, total_q, 1, model_name=args.model, prev_indice=history_indices)


                    # use momentum only to the selected coodinates
                    common_indice = np.intersect1d(indices, prev_indice) #
                    common_indice = common_indice.astype('int32')
                    if args.model == 'wave2letter':
                        tmp_g = np.zeros(shape=spectrograms.size()[1]*spectrograms.size()[2])
                    else:
                        tmp_g = np.zeros(shape=spectrograms.size()[2]*spectrograms.size()[3])

                    tmp_g[common_indice] = prev_g.reshape(-1)[common_indice]
                    tmp_g = tmp_g.reshape(spectrograms.size())
                    g_hat = momentum * tmp_g + (1.0 - momentum) * g_hat
                    prev_g = g_hat
                    prev_indice = indices
                    # print(np.count_nonzero(g_hat))

                elif grad_est == "genetic":
                    total_q += max_query
                    genetic(model, ctc_loss, adv_spec, input_lengths, labels, label_lengths,
                            target_sentence, ori_cer, ori_loss.item(), max_query, samples_per_draw, f,
                            txt_writer, idx, args.B, args.use_TD, model_name=args.model)

                elif grad_est == "zoo":
                    l2_attack(adv_spec, spectrograms, labels, input_lengths, model, ctc_loss, target_sentence,
                              ori_cer, solver, ori_loss.item(), B, evaluation_decoder, f, txt_writer,
                              idx, batch_size=args.samples_per_draw, cw_iter=args.max_query, step_size=step_size,
                              adam_beta1=adam_beta1, adam_beta2=adam_beta2, use_TD=args.use_TD, model_name=args.model)
                    total_q += max_query

                # adv_spec = torch.from_numpy(adv_spec).to(device, dtype=torch.float)
                if grad_est != "zoo" and grad_est != "genetic" and total_q % 1 == 0:
                    # g_hat = np.clip(g_hat, -spectrograms_clip, spectrograms_clip)
                    print(np.mean(g_hat), np.mean(spectrograms_clip))
                    adv_spec = adv_spec + torch.from_numpy(g_hat).to(device, dtype=torch.float)

                    # generate AEs by pertubations
                    if args.model == 'wave2letter':
                        adv_spec = adv_spec.squeeze(dim=1)
                        out = model(adv_spec)
                        out_decode = out.transpose(1, 2)
                        out = out.permute(2, 0, 1)  # TxNxH
                        decoded_output, _ = evaluation_decoder.decode(out_decode, [out_decode.shape[1]])
                    elif args.model == 'ds2':
                        out, output_sizes = model(adv_spec, torch.IntTensor(input_lengths).cuda())
                        decoded_output, _ = evaluation_decoder.decode(out, output_sizes)
                    elif args.model == 'ds2v1':
                        out = model(adv_spec)
                        output = F.log_softmax(out, dim=2)
                        decoded_output, _ = evaluation_decoder.decode(output, torch.IntTensor(input_lengths).cuda())

                    # compute increased CER
                    tar_str = evaluation_decoder.convert_to_strings(labels)
                    adv_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0]) + 0.0001)
                    adv_wer = evaluation_decoder.wer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0].split()) + 0.0001)
                    increase_cer = adv_cer - ori_cer
                    cer_list.append(increase_cer)
                    if increase_cer > cer_criterion and cer_criterion < 1:
                        print("query = {}, ori CER = {:.5f}, adv CER = {:.5f}, increase by = {:.5f}, adv_wer = {:.5f}".
                            format(total_q, ori_cer, adv_cer, increase_cer, adv_wer), file=txt_writer)
                        print("query = {}, ori CER = {:.5f}, adv CER = {:.5f}, increase by = {:.5f}, adv_wer = {:.5f}".
                            format(total_q, ori_cer, adv_cer, increase_cer, adv_wer))
                        cer_criterion += 0.1
                        generated_audios[idx].append(adv_spec.cpu().detach().numpy())
                        if increase_cer >= 0.6:
                            break

                    print("query times: {} ori CER: {} adv CER: {} increase by: {} adv_wer: {}".format(total_q, ori_cer, adv_cer, increase_cer, adv_wer), file=txt_writer)
                    print("query times: {} ori CER: {} adv CER: {} increase by: {} adv_wer: {}".format(total_q, ori_cer, adv_cer, increase_cer, adv_wer))
            pickle.dump(generated_audios, f)
            attack_counts += 1
    print(attack_counts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--samples_per_draw', type=int, required=False, default=200, help='Batch size for generation')
    parser.add_argument('--method', type=str, required=False, default="zoo", help='Method')
    parser.add_argument('--max_query', type=int, required=False, default=500, help='max query')
    parser.add_argument('--B', type=float, required=False, default=0.5, help='strength of noise')
    parser.add_argument('--solver', type=str, required=False, default="adam", help='strength of noise')
    parser.add_argument('--use_TD', type=int, required=False, default=1, help='Whether use temporal dependency')
    parser.add_argument('--dataset', type=str, required=False, default="ls", help='which dataset to use')
    parser.add_argument('--data_path', type=str, required=True, help='which dataset to use')
    parser.add_argument('--model', type=str, required=False, default="ds2", help='which dataset to use')

    args = parser.parse_args()
    main()
