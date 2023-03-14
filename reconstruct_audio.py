import pickle
import pandas as pd
import torch
import torchaudio
from data.LibriSpeech import parse_audio_2, data_processing
from data.LibriSpeech import librispeech_loader
from torchaudio.transforms import GriffinLim
import os
from data.LibriSpeech import librispeech_loader, ted_dataloader
from utils import GreedyDecoder
import argparse



def get_length():
    data_path = "/media/tongch/C8B41AFCB41AED26"
    train_loader, valid_loader, test_loader = ted_dataloader(data_path, batch_size=1, mel_num=128)
    labels = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
              "s", "t", "u", "v", "w", "x", "y", "z", " "]
    evaluation_decoder = GreedyDecoder(labels, blank_index=0)

    count = 0
    for idx, i in enumerate(test_loader):
        if count >= 20:
            break
        spectrograms, labels, input_lengths, label_lengths = i


        tar_str = evaluation_decoder.convert_to_strings(labels)
        if tar_str[0][0] == "ignoretimesegmentinscoring ":
            continue
        else:
            count += 1




def recon(original_audio_info, adv_audio):
    spec, mean, std, phase = parse_audio_2(original_audio_info[0])
    spec = spec.cuda()
    mean = mean.cuda()
    std = std.cuda()
    spec = spec * std + mean
    count = 0

    spec_adv = adv_audio
    spec_adv = torch.from_numpy(spec_adv).cuda()

    print(spec_adv.shape, spec.shape)
    spec_adv = spec_adv * std + mean
    spec_adv = spec_adv * std + mean
    # spec = spec.squeeze(dim=0)
    spec_adv = spec_adv.squeeze(dim=0)
    # print(spec.shape, spec_original.shape)

    griff = GriffinLim(n_fft=320, win_length=320, hop_length=160, window_fn=torch.hamming_window, n_iter=64, power=1)
    waveform_ori = griff.forward(spec.cpu())
    waveform_noise = griff.forward(spec_adv.cpu())
    waveform_adv = waveform_noise + waveform_ori
    # waveform_adv = torch.zeros(waveform_ori.shape[0], waveform_ori.shape[1])
    # waveform_adv[:, :waveform_ori.shape[1]] += waveform_ori
    # waveform_adv[:, :waveform_ori.shape[1]] += waveform_noise

    return waveform_adv, waveform_ori



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dataset', type=str, required=False, default='ls', help='Batch size for generation')
    parser.add_argument('--data_path', type=str, required=True, help='Batch size for generation')

    parser.add_argument('--samples_per_draw', type=int, required=False, default=200, help='Batch size for generation')
    parser.add_argument('--method', type=str, required=False, default="nes", help='Method')
    parser.add_argument('--max_query', type=int, required=False, default=500, help='max query')
    parser.add_argument('--B', type=float, required=False, default=1, help='strength of noise')
    parser.add_argument('--solver', type=str, required=False, default="adam", help='strength of noise')
    parser.add_argument('--use_TD', type=int, required=False, default=1, help='Whether use temporal dependency')
    parser.add_argument('--model', type=str, required=False, default="ds2", help='which dataset to use')
    args = parser.parse_args()


# data_path = "/home/tongch/Downloads/LibriSpeech"
    if args.dataset == 'ls':
        test = torchaudio.datasets.LIBRISPEECH(args.data_path, url='test-clean')
    elif args.dataset == 'ted':
        test = torchaudio.datasets.TEDLIUM(root=args.data_path,release="release2",subset="test")

# test_loader = torch.utils.data.DataLoader(test,
#                                           batch_size=1,
#                                           shuffle=False,
#                                           collate_fn=lambda x: data_processing(x, "test", 161))

    adv_audios = []

    # file = open('output/{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.samples_per_draw,
    #                                               args.method, args.solver, args.max_query, args.B, args.use_TD), 'rb')
    # folder = "{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataseft, args.model, args.samples_per_draw,
    #                                               args.method, args.solver, args.max_query, args.B, args.use_TD)
    file = open('audio_noise.pkl', 'rb')
    folder = 'noise'

    if not os.path.exists(os.path.join("generated_audios", folder)):
        os.mkdir(os.path.join("generated_audios", folder))
    while True:
        data = pickle.load(file)
    # try:
        for idx, audios in data.items():
            print(idx, len(audios))
            original_audio_info = test[idx]

            for i in range(len(audios)):
                adv_audio = audios[min(i, len(audios))]
                recon_adv_audio, recon_ori = recon(original_audio_info, adv_audio)
                torchaudio.save('generated_audios/{}/{}-original.flac'.format(folder, idx), recon_ori, 16000)
                torchaudio.save('generated_audios/{}/{}-{}.flac'.format(folder, idx, i), recon_adv_audio, 16000)