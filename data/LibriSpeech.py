"""Load Lribrispeech data"""
import math
import torchaudio
import torch
import torch.nn as nn
import numpy as np
import re

torch.manual_seed(0)
np.random.seed(42)


# text label pre-processing
class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self, map_type=1):
        self.map_type = map_type
        if map_type == 1:
            char_map_str = """
            _ 0
            ' 1
            a 2
            b 3
            c 4
            d 5
            e 6
            f 7
            g 8
            h 9
            i 10
            j 11
            k 12
            l 13
            m 14
            n 15
            o 16
            p 17
            q 18
            r 19
            s 20
            t 21
            u 22
            v 23
            w 24
            x 25
            y 26
            z 27
            <SPACE> 28
            """
        elif map_type == 2:
            char_map_str = """
            _ 0
            a 1
            b 2
            c 3
            d 4
            e 5
            f 6
            g 7
            h 8
            i 9
            j 10
            k 11
            l 12
            m 13
            n 14
            o 15
            p 16
            q 17
            r 18
            s 19
            t 20
            u 21
            v 22
            w 23
            x 24
            y 25
            z 26
            ' 27
            | 28
            """
        # char_map_str = """
        #         sos_id 0
        #         eos_id 1
        #         ' 2
        #         <SPACE> 3
        #         a 4
        #         b 5
        #         c 6
        #         d 7
        #         e 8
        #         f 9
        #         g 10
        #         h 11
        #         i 12
        #         j 13
        #         k 14
        #         l 15
        #         m 16
        #         n 17
        #         o 18
        #         p 19
        #         q 20
        #         r 21
        #         s 22
        #         t 23
        #         u 24
        #         v 25
        #         w 26
        #         x 27
        #         y 28
        #         z 29
        #         """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        if map_type == 1:
            self.index_map[28] = " "
        elif map_type == 2:
            self.index_map[0] = " "

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                if self.map_type == 1:
                    ch = self.char_map["<SPACE>"]
                elif self.map_type == 2:
                    ch = self.char_map["_"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        if self.map_type == 1:
            return "".join(string).replace("<SPACE>", " ").replace("eos_id", "").replace("sos_id","")
        elif self.map_type == 2:
            return "".join(string).replace("|", " ").replace("eos_id", "").replace("sos_id","")


def parse_audio_2(waveform, sample_rate=16000, window_size=0.02, window_stride=0.01):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    D = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=torch.hamming_window(320))
    spect, phase = torchaudio.functional.magphase(D)
    # D = librosa.stft(waveform.squeeze().numpy(), n_fft=n_fft, hop_length=hop_length,
    #                  win_length=win_length, window='hamming')
    # spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)

    mean = spect.mean()
    std = spect.std()
    spect.add_(-mean)
    spect.div_(std)

    return spect, mean, std, phase

def parse_audio(waveform, sample_rate=16000, window_size=0.02, window_stride=0.01):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    D = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=torch.hamming_window(320))
    spect, phase = torchaudio.functional.magphase(D)
    # D = librosa.stft(waveform.squeeze().numpy(), n_fft=n_fft, hop_length=hop_length,
    #                  win_length=win_length, window='hamming')
    # spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)

    mean = spect.mean()
    std = spect.std()
    spect.add_(-mean)
    spect.div_(std)

    return spect


def frontend(signal, sample_rate=16000, nfft=512, nfilt=64, preemph=0.97, window_size=0.020, window_stride=0.010):
    def get_melscale_filterbanks(nfilt, nfft, samplerate):
        hz2mel = lambda hz: 2595 * math.log10(1 + hz / 700.)
        mel2hz = lambda mel: torch.mul(700, torch.sub(torch.pow(10, torch.div(mel, 2595)), 1))

        lowfreq = 0
        highfreq = samplerate // 2
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        melpoints = torch.linspace(lowmel, highmel, nfilt + 2)
        bin = torch.floor(torch.mul(nfft + 1, torch.div(mel2hz(melpoints), samplerate))).tolist()

        fbank = torch.zeros([nfilt, nfft // 2 + 1]).tolist()
        for j in range(nfilt):
            for i in range(int(bin[j]), int(bin[j + 1])):
                fbank[j][i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j][i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return torch.tensor(fbank)

    preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
    win_length = int(sample_rate * (window_size + 1e-8))
    hop_length = int(sample_rate * (window_stride + 1e-8))
    pspec = torch.stft(preemphasis(signal, preemph), nfft, win_length=win_length, hop_length=hop_length,
                       window=torch.hann_window(win_length), pad_mode='constant', center=False).pow(2).sum(dim=-1) / nfft
    mel_basis = get_melscale_filterbanks(nfilt, nfft, sample_rate).type_as(pspec)
    features = torch.log(torch.add(torch.matmul(mel_basis, pspec), 1e-20))
    return (features - features.mean()) / features.std()

def remove_punc(s):
    regex = re.compile('[^a-zA-Z\'_]')
    s_re = regex.sub(' ', s)
    s_re = s_re.replace("unk","")
    return s_re

def fbank_processing(data):
    features = []
    labels = []
    input_lengths = []
    label_lengths = []
    text_transform = TextTransform(map_type=2)
    for batch in data:
        waveform = batch[0]

        utterance = batch[2]
        waveform = waveform.squeeze()
        feature = frontend(waveform).transpose(0, 1)
        features.append(feature)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(feature.shape[0])
        label_lengths.append(len(label))

    features = (
        nn.utils.rnn.pad_sequence(features, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    features = features.squeeze(dim=1)

    return features, labels, input_lengths, label_lengths


def data_processing(data,loader,mel_num, model_name='ds2'):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    if loader == "train":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=mel_num),
            torchaudio.transforms.Spectrogram(n_fft=320),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
    if loader == "test":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=mel_num),
            # torchaudio.transforms.Spectrogram(n_fft=320)
        )
    text_transform = TextTransform()

    for (waveform, _, utterance, _, _, _) in data:
        if model_name == 'ds2v1':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif model_name == 'ds2':
            spec = parse_audio(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        utterance = remove_punc(utterance)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0])
        # input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def librispeech_loader(path,batch_size,mel_num,process_type=1, model_name='ds2'):
    train = torchaudio.datasets.LIBRISPEECH(path, url='test-clean')
    valid = torchaudio.datasets.LIBRISPEECH(path, url='test-clean')
    test = torchaudio.datasets.LIBRISPEECH(path, url='test-clean')
    # for i in test:
    #     _,mean,std,phase = parse_audio_2(i[0])

    # print(test[0])
    # def load_and_cache_examples(args, tokenizer, evaluate=False, fpath=None):
    #     if fpath:
    #         dataset = TextDataset(tokenizer, args, fpath)
    #     else:
    #         dataset = TextDataset(tokenizer, args, args.eval_data_path if evaluate else args.train_data_path)
    #
    #     # Ignore incomplete batches
    #     # If you don't do this, you'll get an error at the end of training
    #     n = len(dataset) % args.per_gpu_train_batch_size
    #     if n != 0:
    #         dataset.examples = dataset.examples[:-n]
    #     return dataset

    if process_type == 1:
        train_loader = torch.utils.data.DataLoader(train,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=lambda x:data_processing(x,"train",mel_num, model_name))

        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   collate_fn=lambda x: data_processing(x, "test",mel_num, model_name))
        test_loader = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=lambda x: data_processing(x,"test",mel_num, model_name))

    elif process_type == 2:
        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x:fbank_processing(x))

        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   collate_fn=lambda x:fbank_processing(x))
        test_loader = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=lambda x:fbank_processing(x))
    return train_loader,valid_loader, test_loader


def ted_dataloader(path,batch_size,mel_num,process_type=1):
    train = torchaudio.datasets.TEDLIUM(root=path,release="release3")
    valid = torchaudio.datasets.TEDLIUM(root=path,release="release2",subset="dev")
    test = torchaudio.datasets.TEDLIUM(root=path,release="release2",subset="test")

    if process_type == 1:
        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x: data_processing(x, "train", mel_num))

        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   collate_fn=lambda x: data_processing(x, "test", mel_num))
        test_loader = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=lambda x: data_processing(x, "test", mel_num))

    elif process_type == 2:
        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x: fbank_processing(x))

        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   collate_fn=lambda x: fbank_processing(x))
        test_loader = torch.utils.data.DataLoader(test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=lambda x: fbank_processing(x))
    return train_loader, valid_loader, test_loader




