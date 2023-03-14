from Levenshtein import distance
from jiwer import wer
import numpy as np
import torch
import torch.nn.functional as F
# import librosa
# import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from data.LibriSpeech import TextTransform
import random
import math
# import warnings
# warnings.simplefilter('ignore')
np.random.seed(1)
torch.manual_seed(1)


def char_error(tar, pred):
    return round(distance(tar, pred)/len(tar),4)


def word_error(tar, pred):
    return wer(tar, pred)


def snr(signal, noise):
    """Signal to noise ratio"""
    sig = np.sum(signal ** 2)
    noi = np.sum(noise ** 2)
    snr_db = 10 * np.log10(sig/noi)
    return snr_db


# def draw_spectrogram(melspec,snr_db):
#     librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max),y_axis='mel',sr=16000)
#     plt.title('Mel spectrogram, SNR %.4f' % (snr_db))
#     plt.colorbar(format="%+2.f dB")
#     plt.show()

def get_cossim_2(benign,ae, h,k):
    benign = np.squeeze(benign)
    ae = np.squeeze(ae)
    benign_corr, ae_corr, diff = [], [], []
    s = 0

    # print(mel_spec)
    sim = []
    for s in range(0, ae.shape[1] - 3):
        t1_b = benign[:, s:s + h]
        t1_ae = ae[:, s:s + h]
        t2_b = benign[:, s+1:s + h + 1]
        t2_ae = ae[:, s+1:s + h + 1]
        cs_b = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        cs_ae = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        sim.append(cs_ae[0][0]-t1_ae[0][0])
        # pr_b = pearsonr(t1_b.flatten().tolist(), t2_b.flatten().tolist())
        # pr_ae = pearsonr(t1_ae.flatten().tolist(), t2_ae.flatten().tolist())

        # diff.append(float(pr_b[0] - pr_ae[0]))
        # benign_corr.append(float(pr_b[0]))
        # ad_corr.append(float(pr_ae[0]))
        # diff.append(float(cs_b - cs_ae))
        # benign_corr.append(float(cs_b[0]))
        # ae_corr.append(float(cs_ae[0]))

    # select k max index with the lowest similarity
    diff_logits = np.exp(np.abs(np.array(diff)) - np.abs(np.array(diff)).max())
    diff_npy = np.abs(np.array(diff)) *(-1)
    _diff_logits = (diff_npy - np.min(diff_npy))/(np.max(diff_npy) - np.min(diff_npy))
    _diff_logits = diff_logits
    diff_probs = _diff_logits / _diff_logits.sum()

    ind_kmax = np.random.choice(len(diff_probs), p=diff_probs,size=k)

    # ind_max = np.argsort(np.array(sim))
    # ind_kmax = ind_max[:10]
    # print(ind_kmax)
    # print(ind_kmax)
    return ind_kmax


def get_cossim(benign,ae, h,k):
    benign = np.squeeze(benign)
    ae = np.squeeze(ae)
    benign_corr, ae_corr, diff = [], [], []
    s = 0
    # print(mel_spec)

    for s in range(0, ae.shape[1] - 2):
        t1_b = benign[:, s:s + h]
        t1_ae = ae[:, s:s + h]
        t2_b = benign[:, s:s + h]
        t2_ae = ae[:, s:s + h]
        if type(t1_b) == torch.Tensor:
            t1_b = t1_b.detach().cpu().numpy()
        if type(t1_ae) == torch.Tensor:
            t1_ae = t1_ae.detach().cpu().numpy()
        if type(t2_b) == torch.Tensor:
            t2_b = t2_b.detach().cpu().numpy()
        if type(t2_ae) == torch.Tensor:
            t2_ae = t2_ae.detach().cpu().numpy()
        cs_b = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        cs_ae = cosine_similarity([t1_ae.ravel().tolist()], [t2_ae.ravel().tolist()])
        # pr_b = pearsonr(t1_b.flatten().tolist(), t2_b.flatten().tolist())
        # pr_ae = pearsonr(t1_ae.flatten().tolist(), t2_ae.flatten().tolist())

        # diff.append(float(pr_b[0] - pr_ae[0]))
        # benign_corr.append(float(pr_b[0]))
        # ad_corr.append(float(pr_ae[0]))
        diff.append(float(cs_b - cs_ae))
        benign_corr.append(float(cs_b[0]))
        ae_corr.append(float(cs_ae[0]))

    # select k max index with the lowest similarity
    diff_logits = np.exp(np.abs(np.array(diff)) - np.abs(np.array(diff)).max())
    _diff_logits = (-1) * diff_logits
    diff_probs = _diff_logits / _diff_logits.sum()


    # ind_kmax = np.random.choice(len(diff_probs), p=diff_probs,size=k)

    ind_max = np.argsort(np.abs(np.array(diff)))
    ind_kmax = ind_max[:k]
    return ind_kmax


def get_cossim_3(benign,ae, h,k):
    benign = np.squeeze(benign)
    ae = np.squeeze(ae)
    benign_corr, ae_corr, diff = [], [], []
    s = 0

    # print(mel_spec)

    for s in range(0, ae.shape[-1] - 3):
        t1_b = benign[:, s:s + h]
        t1_ae = ae[:, s:s + h]
        t2_b = benign[:, s+1:s + h + 1]
        t2_ae = ae[:, s+1:s + h + 1]
        cs_b = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        cs_ae = cosine_similarity([t1_ae.ravel().tolist()], [t2_ae.ravel().tolist()])
        # pr_b = pearsonr(t1_b.flatten().tolist(), t2_b.flatten().tolist())
        # pr_ae = pearsonr(t1_ae.flatten().tolist(), t2_ae.flatten().tolist())

        # diff.append(float(pr_b[0] - pr_ae[0]))
        # benign_corr.append(float(pr_b[0]))
        # ad_corr.append(float(pr_ae[0]))
        diff.append(float(cs_b - cs_ae))
        benign_corr.append(float(cs_b[0]))
        ae_corr.append(float(cs_ae[0]))

    # select k max index with the lowest similarity
    # diff_logits = np.exp(np.abs(np.array(diff)) - np.abs(np.array(diff)).max())
    diff_npy = np.abs(np.array(diff)) *(-1)
    _diff_logits = (diff_npy - np.min(diff_npy))/(np.max(diff_npy) - np.min(diff_npy))
    # _diff_logits = diff_logits
    diff_probs = _diff_logits / _diff_logits.sum()

    ind_kmax = np.random.choice(len(diff_probs), p=diff_probs,size=k)

    # ind_max = np.argsort(np.abs(np.array(diff)))
    # ind_kmax = ind_max[:10]
    # print(ind_kmax)
    return ind_kmax

def td(adv, spe, i_len, w_width,batch_size, history_indices):
    """compute the similarity between windows, and find the frames with the highest """
    # coord_no = torch.squeeze(spectrograms).cpu().detach().numpy().shape[1] / 5
    # coord_no = 4
    # h = 3  # sliding window steps
    # idx = get_cossim(adv, spe, w_width, int(math.ceil(batch_size/161)))
    idx = get_cossim(adv, spe, w_width, int(batch_size))
    var = adv.reshape(-1)
    # find corresponded col in spectrum
    # idx_c = [w_width + i for i in idx]
    idx_c = idx
    indice = []
    idx_all = []
    # points_per_col = batch_size // len(idx_c)
    for i in idx_c:
        idx_col = np.linspace(i, (adv.size(-2)-1) * i_len + i, adv.size(-2), dtype=int)
        # indice.append(idx_col[torch.argmax(var[idx_col])])
        # indice.append(np.random.choice(idx_col, points_per_col, replace=False))
        idx_all.append(idx_col.tolist())

    idx_all = [item for sublist in idx_all for item in sublist]
    # idx_all = []
    # current_idx = []

    # for sublist in idx_all:
    #     for item in sublist:
    #         if item not in history_indices:
    #             current_idx.append(item)
    current_idx = list(set(idx_all).difference(set(history_indices)))

    if len(current_idx) < batch_size:
        current_idx = idx_all

    indice = np.random.choice(current_idx,size= batch_size, replace=False)
    # indice = np.concatenate(indice)
    # print(indice)
    return idx_c, idx_all,indice

def draw_cossim(benign,ae, h, snr_db, idx):
    benign_corr, ae_corr, diff = [], [], []
    s = 0
    # print(mel_spec)
    t1_b = benign[:, s:s + h]
    t1_ae = ae[:, s:s + h]
    for s in range(1, ae.shape[1] - 2):
        t2_b = benign[:, s:s + h]
        t2_ae = ae[:, s:s + h]
        cs_b = cosine_similarity(t1_b.flatten().reshape(1,-1),t2_b.flatten().reshape(1,-1))
        cs_ae = cosine_similarity(t1_ae.flatten().reshape(1,-1),t2_ae.flatten().reshape(1,-1))
        # pr_b = pearsonr(t1_b.flatten().tolist(), t2_b.flatten().tolist())
        # pr_ae = pearsonr(t1_ae.flatten().tolist(), t2_ae.flatten().tolist())

        # diff.append(float(pr_b[0] - pr_ae[0]))
        # benign_corr.append(float(pr_b[0]))
        # ad_corr.append(float(pr_ae[0]))
        diff.append(float(cs_b-cs_ae))
        benign_corr.append(float(cs_b[0]))
        ae_corr.append(float(cs_ae[0]))
        t1_b = t2_b
        t1_ae = t2_ae

    # draw
    plt.plot(benign_corr, label="benign", alpha=0.4)
    plt.plot(ae_corr, label="adversarial", alpha=0.4)
    plt.plot(diff, label="diff", alpha=0.7)
    plt.title("SNR=%.2f, audio %d" % (snr_db, idx+1))
    plt.legend()
    plt.show()

### ds2_v1 decoder
def deepspeech_decoder(output,libris_transform):
    output = F.log_softmax(output, dim=2)
    arg_maxes = torch.argmax(output, dim=2)
    res = [j.item() for j in arg_maxes[0]]
    pred_output = greedydecoder(libris_transform.int_to_text(res))
    return pred_output

def greedydecoder(res):
    pred_output = ""
    i, j = 0, 0
    while i < len(res):
        if res[j] == res[i]:
            i += 1
        else:
            pred_output = pred_output + res[j]
            j = i
            i += 1
    return pred_output


### ds2_v2 decoder
class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets


# wav2letter decoder
def W2lGreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    # print(output_latest_ds2v1.shape)
    text_transform = TextTransform()
    arg_maxes = torch.argmax(output, dim=2) #[1,639]
    # print(arg_maxes)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][: label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets