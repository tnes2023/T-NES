# https://github.com/MyrtleSoftware/deepspeech/blob/master/src/deepspeech/run.py

from torch import nn


class OverLastDim(nn.Module):
    """Collapses a tensor to 2D, applies a module, and (re-)expands the tensor.

    An n-dimensional tensor of shape (s_1, s_2, ..., s_n) is first collapsed to
    a tensor with shape (s_1*s_2*...*s_n-1, s_n). The module is called with
    this as input producing (s_1*s_2*...*s_n-1, s_n') --- note that the final
    dimension can change. This is expanded to (s_1, s_2, ..., s_n-1, s_n') and
    returned.

    Args:
        module (nn.Module): Module to apply. Must accept a 2D tensor as input
            and produce a 2D tensor as output_latest_ds2v1, optionally changing the size of
            the last dimension.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        *dims, input_size = x.size()

        reduced_dims = 1
        for dim in dims:
            reduced_dims *= dim

        x = x.reshape(reduced_dims, -1)
        x = self.module(x)
        x = x.reshape(*dims, -1)
        return x

class Network(nn.Module):
    """A network with 3 FC layers, a Bi-LSTM, and 2 FC layers.

    Args:
        in_features: Number of input features per step per batch.
        n_hidden: Internal hidden unit size.
        out_features: Number of output_latest_ds2v1 features per step per batch.
        drop_prob: Dropout drop probability.
        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
    """

    def __init__(self, in_features, n_hidden=2048, out_features=29, drop_prob=0.25,
                 relu_clip=20.0, forget_gate_bias=1.0):
        super().__init__()

        self._relu_clip = relu_clip
        self._drop_prob = drop_prob

        self.fc1 = self._fully_connected(in_features, n_hidden)
        self.fc2 = self._fully_connected(n_hidden, n_hidden)
        self.fc3 = self._fully_connected(n_hidden, 2*n_hidden)
        self.bi_lstm = self._bi_lstm(2*n_hidden, n_hidden, forget_gate_bias)
        self.fc4 = self._fully_connected(2*n_hidden, n_hidden)
        self.out = self._fully_connected(n_hidden,
                                         out_features,
                                         relu=False,
                                         dropout=False)

    def _fully_connected(self, in_f, out_f, relu=True, dropout=True):
        layers = [nn.Linear(in_f, out_f)]
        if relu:
            layers.append(nn.Hardtanh(0, self._relu_clip, inplace=True))
        if dropout:
            layers.append(nn.Dropout(p=self._drop_prob))
        return OverLastDim(nn.Sequential(*layers))

    def _bi_lstm(self, input_size, hidden_size, forget_gate_bias):
        lstm = nn.LSTM(input_size=input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
        if forget_gate_bias is not None:
            for name in ['bias_ih_l0', 'bias_ih_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
            for name in ['bias_hh_l0', 'bias_hh_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(0)
        return lstm

    def forward(self, x):
        """Computes a single forward pass through the network.

        Args:
            x: A tensor of shape (seq_len, batch, in_features).

        Returns:
            Logits of shape (seq_len, batch, out_features).
        """
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        h, _ = self.bi_lstm(h)
        h = self.fc4(h)
        out = self.out(h)
        return out
#
# class DeepSpeech(nn.Module):
#     """Deep Speech Model.
#
#     Args:
#         optimiser_cls: See `Model`.
#         optimiser_kwargs: See `Model`.
#         decoder_cls: See `Model`.
#         decoder_kwargs: See `Model`.
#         n_hidden (int): Internal hidden unit size.
#         n_context (int): Number of context frames to use on each side of the
#             current input frame.
#         n_mfcc (int): Number of Mel-Frequency Cepstral Coefficients to use as
#             input for a single frame.
#         drop_prob (float): Dropout drop probability, [0.0, 1.0] inclusive.
#         winlen (float): Window length in ms to compute input features over.
#         winstep (float): Window step size in ms.
#         sample_rate (int): Sample rate in Hz of input data.
#
#     Attributes:
#         See base class.
#     """
#
#     def __init__(self, optimiser_cls=None, optimiser_kwargs=None,
#                  decoder_cls=None, decoder_kwargs=None,
#                  n_hidden=2048, n_context=9, n_mfcc=26, drop_prob=0.25,
#                  winlen=0.025, winstep=0.02, sample_rate=16000):
#
#         self._n_hidden = n_hidden
#         self._n_context = n_context
#         self._n_mfcc = n_mfcc
#         self._drop_prob = drop_prob
#         self._winlen = winlen
#         self._winstep = winstep
#         self._sample_rate = sample_rate
#
#         network = self._get_network()
#
#         # super().__init__(network=network,
#         #                  optimiser_cls=optimiser_cls,
#         #                  optimiser_kwargs=optimiser_kwargs,
#         #                  decoder_cls=decoder_cls,
#         #                  decoder_kwargs=decoder_kwargs,
#         #                  clip_gradients=None)
#
#     def _get_network(self):
#         return Network(in_features=self._n_mfcc*(2*self._n_context + 1),
#                        n_hidden=self._n_hidden,
#                        out_features=len(self.ALPHABET),
#                        drop_prob=self._drop_prob)
#
    # @property
    # def transform(self):
    #     return Compose([preprocess.MFCC(self._n_mfcc),
    #                     preprocess.AddContextFrames(self._n_context),
    #                     preprocess.Normalize(),
    #                     torch.FloatTensor,
    #                     lambda t: (t, len(t))])
    #
    # @property
    # def target_transform(self):
    #     return Compose([str.lower,
    #                     self.ALPHABET.get_indices,
    #                     torch.IntTensor])
