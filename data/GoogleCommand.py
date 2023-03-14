import torch
import torchaudio
import torch.nn as nn
import numpy as np
from data.LibriSpeech import fbank_processing


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


def data_processing(data,loader,num_mel):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    if loader == "train":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=num_mel),
            # torchaudio.transforms.Spectrogram(),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
    if loader == "test":
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=num_mel),
            # torchaudio.transforms.Spectrogram()
        )
    text_transform = TextTransform()

    for waveform, sample_rate, label, speaker_id, utterance_number in data:
        # spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spec = parse_audio(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        utterance = label
        # print(utterance)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
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



def gs_dataloader(path,batch_size,num_mel,process_type=1):
    train = torchaudio.datasets.SPEECHCOMMANDS(root=path,url="speech_commands_v0.01", subset="training")
    valid = torchaudio.datasets.SPEECHCOMMANDS(root=path,url="speech_commands_v0.01", subset="validation")
    test = torchaudio.datasets.SPEECHCOMMANDS(root=path,url="speech_commands_v0.01", subset="testing")

    if process_type == 1:
        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x: data_processing(x, "train",num_mel))

        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   collate_fn=lambda x: data_processing(x, "test",num_mel))
        test_loader = torch.utils.data.DataLoader(test,
                                           batch_size=1,
                                           shuffle=False,
                                           collate_fn=lambda x: data_processing(x, "test",num_mel))
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
