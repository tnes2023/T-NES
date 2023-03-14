from pypesq import pesq

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
import soundfile as sf


np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    if not os.path.exists('output/'):
        os.mkdir('output')
    f = open("audio_noise.pkl", 'wb')
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


    if args.model == 'wave2letter': # decoder for Wav2letter
        evaluation_decoder = GreedyDecoder(labels, blank_index=28)
    else: # decoder for other models
        evaluation_decoder = GreedyDecoder(labels, blank_index=0)

    # f = open("output/{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(args.dataset, args.model, args.samples_per_draw,
    #                                               args.method, args.solver, args.max_query, B, args.use_TD), 'wb')
    # txt_writer = open("output/{}_{}_{}_{}_{}_{}_{}_{}.txt".format(args.dataset, args.model, args.samples_per_draw,
    #                                               args.method, args.solver, args.max_query, B, args.use_TD), 'w')
    txt_writer = open('noise_exp_2.txt', 'w')
    change_list = []
    attack_counts = 0
    generated_audios = {}

    for idx, i in enumerate(test_loader):
        history_indices = []
        cer_list = []
        cer_criterion = 0.1
        # if idx >= 0 and idx < 20:
        txt_writer.write(str(idx) + ' \n')
        if attack_counts < 1:
            # print("==============================", file=txt_writer)
            # print("ID:", idx + 1, file=txt_writer)

            spectrograms, labels, input_lengths, label_lengths = i
            # sf.write('original.wav', spectrograms.squeeze(), 16000)

            # print(spectrograms)
            spectrograms, labels = spectrograms.to("cuda", dtype=torch.float), labels.to("cuda", dtype=torch.float)

            # spectrograms_clip = spectrograms * B # set constraints for esitmated gradients
            # spectrograms_clip = spectrograms_clip.detach().cpu().numpy()
            # adv_spec = spectrograms.detach()

            decoded_output = run_model(model, args.model, spectrograms, ctc_loss, labels, input_lengths, label_lengths, evaluation_decoder)

            # ======== predicted sentence by DeepSpeech(character string) ========
            cer, wer = evaluate_result(decoded_output, evaluation_decoder, labels)
            if cer is None:
                continue


            txt_writer.write('original audio: cer {}, wer {}\n'.format(cer, wer))
            # tar_str = evaluation_decoder.convert_to_strings(labels)
            # if tar_str[0][0] == "ignoretimesegmentinscoring ": # pass audios in TED
            #     print(tar_str)
            #     continue
            #
            # # compute original cer and wer
            # target_sentence = tar_str[0][0]
            # ori_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0]) + 0.0001)
            # ori_wer = evaluation_decoder.wer(decoded_output[0][0], tar_str[0][0]) / (len(decoded_output[0][0].split(' ')) + 0.0001)
            # print("Decoded Output:", decoded_output)
            # print("Target Output: ", tar_str)
            # print("CER:", ori_cer, "WER:", ori_wer)

            # add gaussian noise on the original audio
            std = 1
            mean = torch.mean(spectrograms).cuda()
            print('audio mean: ', mean)
            for j in range(10, 20):
                audio_noise = spectrograms + torch.randn_like(spectrograms).cuda() * mean * j * 10
                decoded_output_noise = run_model(model, args.model, audio_noise, ctc_loss, labels, input_lengths, label_lengths, evaluation_decoder)
                cer, wer = evaluate_result(decoded_output_noise, evaluation_decoder, labels)
                txt_writer.write(
                    'noise audio {}: cer {}, wer {}\n'.format(j+1, cer, wer))

                if idx not in generated_audios:
                    generated_audios[idx] = [audio_noise.cpu().detach().numpy()]
                else:
                    generated_audios[idx].append(audio_noise.cpu().detach().numpy())


            # sf.write('noise.wav', audio_noise, 16000)
            attack_counts += 1

            # break

    pickle.dump(generated_audios, f)

def evaluate_result(decoded_output, evaluation_decoder, labels):
    tar_str = evaluation_decoder.convert_to_strings(labels)
    if tar_str[0][0] == "ignoretimesegmentinscoring ":  # pass audios in TED
        print('wrong input: ', tar_str)
        # continue
        return None, None

    # compute original cer and wer
    target_sentence = tar_str[0][0]
    ori_cer = evaluation_decoder.cer(decoded_output[0][0], tar_str[0][0]) / (
                len(decoded_output[0][0]) + 0.0001)
    ori_wer = evaluation_decoder.wer(decoded_output[0][0], tar_str[0][0]) / (
                len(decoded_output[0][0].split(' ')) + 0.0001)
    print("Decoded Output:", decoded_output)
    print("Target Output: ", tar_str)
    print("CER:", ori_cer, "WER:", ori_wer)
    return ori_wer, ori_wer


def run_model(model, model_name, spectrograms, ctc_loss, labels, input_lengths, label_lengths, evaluation_decoder):
    if model_name == 'wave2letter':
        out = model(spectrograms)
        out_decode = out.clone()
        out_decode = out_decode.transpose(1, 2)
        out = out.permute(2, 0, 1)  # TxNxH
        out = out.log_softmax(-1)
        float_out = out.float()  # ensure float32 for loss
        ori_loss = ctc_loss(float_out, labels,
                            torch.IntTensor([float_out.shape[0]] * len(
                                input_lengths)).cuda(),
                            torch.IntTensor(label_lengths).cuda())

        decoded_output, _ = evaluation_decoder.decode(out_decode,
                                                      [out_decode.shape[1]])


    elif model_name == 'ds2':
        out, output_sizes = model(spectrograms,
                                  torch.IntTensor(input_lengths).cuda())
        out_ctc = out.transpose(0, 1)
        out_ctc = out_ctc.log_softmax(-1)
        ori_loss = ctc_loss(out_ctc, labels, output_sizes,
                            torch.IntTensor(label_lengths).cuda())

        decoded_output, _ = evaluation_decoder.decode(out, output_sizes)
    elif model_name == 'ds2v1':
        out = model(spectrograms)
        output = F.log_softmax(out, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        input_lengths = [length // 2 for length in input_lengths]
        ori_loss = ctc_loss(output,
                            labels,
                            torch.IntTensor(input_lengths).cuda(),
                            torch.IntTensor(label_lengths).cuda())

        decoded_output, _ = evaluation_decoder.decode(output.transpose(0, 1),
                                                      torch.IntTensor(
                                                          input_lengths).cuda())

    return decoded_output


def compare_audio():
    audio_ori, sr_ori = sf.read('generated_audios/ls_ds2_200_nes_adam_500_1_1/0-original.flac')
    audio_attack, sr_attack = sf.read('generated_audios/ls_ds2_200_nes_adam_500_1_1/0-5.flac')
    audio_noise, sr_noise = sf.read('generated_audios/noise/0-9.flac')
    # print(audio_attack[:100])
    # print(audio_ori[:100])
    print('mse: ', np.mean((audio_ori - audio_attack)**2))
    print('mse: ', np.mean((audio_ori - audio_noise)**2))

    score_attack = pesq(audio_ori, audio_attack, sr_ori)
    score_noise = pesq(audio_ori, audio_noise, sr_ori)

    print('pesq score: ', score_attack, score_noise)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('--samples_per_draw', type=int, required=False, default=200, help='Batch size for generation')
    # parser.add_argument('--method', type=str, required=False, default="nes", help='Method')
    # parser.add_argument('--max_query', type=int, required=False, default=500, help='max query')
    # parser.add_argument('--B', type=float, required=False, default=0.5, help='strength of noise')
    # parser.add_argument('--solver', type=str, required=False, default="adam", help='strength of noise')
    # parser.add_argument('--use_TD', type=int, required=False, default=1, help='Whether use temporal dependency')

    # parser.add_argument('--dataset', type=str, required=False, default="ls", help='which dataset to use')
    # parser.add_argument('--data_path', type=str, required=True, help='which dataset to use')
    # parser.add_argument('--model', type=str, required=False, default="ds2", help='which dataset to use')
    # #
    # args = parser.parse_args()
    # main()
    compare_audio()