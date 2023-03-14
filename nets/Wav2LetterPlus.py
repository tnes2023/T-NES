import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(kernel_size, num_channels, stride=1, dilation=1, repeat=1, padding=0):
    modules = []
    for i in range(repeat):
        modules.append(
            nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding=padding))
        modules.append(nn.Hardtanh(0, 20, inplace=True))
    return nn.Sequential(*modules)


class Wav2LetterPlus(nn.Module):
    def __init__(self, num_classes):
        super(Wav2LetterPlus, self).__init__()
        self.model = nn.Sequential(
            conv_block(kernel_size=11, num_channels=(64, 256), stride=2, padding=5),
            conv_block(kernel_size=11, num_channels=(256, 256), repeat=3, padding=5),
            conv_block(kernel_size=13, num_channels=(256, 384), repeat=3, padding=6),
            conv_block(kernel_size=17, num_channels=(384, 512), repeat=3, padding=8),
            conv_block(kernel_size=21, num_channels=(512, 640), repeat=3, padding=10),
            conv_block(kernel_size=25, num_channels=(640, 768), repeat=3, padding=12),
            conv_block(kernel_size=29, num_channels=(768, 896), repeat=1, padding=28, dilation=2),
            conv_block(kernel_size=1, num_channels=(896, 1024), repeat=1),
            nn.Conv1d(1024, num_classes, 1)
        )

    def forward(self, x):
        y = self.model(x)
        return y

if __name__ == '__main__':
    model = Wav2LetterPlus(29)
    model.load_state_dict(torch.load('../ckpt/wav2letter_plus.pt'))
    # model = torch.load('../ckpt/wav2letter_plus.pth')
    print(model)
