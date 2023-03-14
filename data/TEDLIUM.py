import torch
import torchaudio
import torch.nn as nn
import numpy as np
import string
import re

torch.manual_seed(42)

# text label pre-processing
class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
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

        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[28] = " "

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")

def remove_punc(s):
    regex = re.compile('[^a-zA-Z\'_]')
    s_re = regex.sub(' ', s)
    s_re = s_re.replace("unk","")
    return s_re

def data_processing(data,loader,mel_num):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    if loader == "train":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=mel_num),
            # torchaudio.transforms.Spectrogram(),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
    if loader == "test":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=mel_num),
            # torchaudio.transforms.Spectrogram()
        )
    text_transform = TextTransform()

    for (waveform, sample_rate, transcript, talk_id, speaker_id, identifier) in data:
        # print("===\n",transcript)
        spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        utterance = remove_punc(transcript).strip()
        # print(utterance)
        label = torch.IntTensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        # input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def ted_dataloader(path,batch_size,mel_num):
    train = torchaudio.datasets.TEDLIUM(root=path,release="release3")
    valid = torchaudio.datasets.TEDLIUM(root=path,release="release2",subset="dev")
    test = torchaudio.datasets.TEDLIUM(root=path,release="release2",subset="test")

    train_loader = torch.utils.data.DataLoader(train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                               collate_fn=lambda x: data_processing(x, "train",mel_num))
    valid_loader = torch.utils.data.DataLoader(valid,
                                               batch_size=1,
                                               shuffle=False,
                                               collate_fn=lambda x: data_processing(x, "test",mel_num))
    test_loader = torch.utils.data.DataLoader(test,
                                               batch_size=1,
                                               shuffle=False,
                                               collate_fn=lambda x: data_processing(x, "test",mel_num))

    return train_loader, valid_loader, test_loader