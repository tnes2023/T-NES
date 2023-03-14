import numpy as np
import torch
from utils import td,get_cossim
import torch.nn.functional as F
from .td_sim import get_tdsim_indice,td_sim_2
# np.random.seed(42)

def get_grad_batch(model, loss_f, input, input_len, label, label_length, model_name='ds2'):
    if model_name == 'wave2letter':
        y_hat = model(input)
        out = y_hat.permute(2, 0, 1)  # TxNxH
        out = out.log_softmax(-1)
        float_out = out.float()  # ensure float32 for loss
        l = loss_f(float_out, label,
                            torch.IntTensor([float_out.shape[0]] * len(input_len)).cuda(),
                            label_length) # compute ctc loss
    elif model_name == 'ds2':
        y_hat, output_sizes = model(input,input_len)
        out = y_hat.transpose(0, 1)
        out = out.log_softmax(-1)
        l = loss_f(out,label,output_sizes, label_length)
        # l = loss_f()
    elif model_name == 'ds2v1':
        y_hat = model(input)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        l = loss_f(output,
                   label,
                   input_len,
                   label_length)

    return y_hat.cpu().detach().numpy(), l.cpu().detach().numpy()


def nes(model, ori_input, input, loss, label, input_lengths, batch_size, sigma, total_q,use_TD, model_name='ds2', prev_indice=[]):
    """https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/NES.py"""
    g_eval = np.zeros(input.size()).reshape(-1)
    var_len = input.reshape(-1).size()[0]
    var_list = np.array(range(0, var_len), dtype=np.int32)
    indices = var_list[np.random.choice(var_list.size, batch_size, replace=False)] # random select indices
    indices = np.array(indices)

    if use_TD==1: # select indices by TD
        # indice = get_tdsim_indice(input, batch_size=batch_size)
        h = 3 # window length for TD
        # if total_q <= 3:  # initalize
        #     var_list = np.array(range(0, input.reshape(-1).size()[0]), dtype=np.int32)
        #     indices = var_list[np.random.choice(var_list.size, int(batch_size), replace=False)]  # selected coord index
        # else:  # after the first iteration
        _, _, indice = td(ori_input, input, input.size()[-1], h, batch_size, prev_indice)
        prev_indice += indice.tolist()
        indices = np.array(indice)


    ## generate evaluation points for NES
    noise_pos = np.random.normal(size=input.size())
    noise = np.concatenate([noise_pos, -noise_pos], axis=0)
    eval_points = input.cpu().detach().numpy() + sigma * noise
    eval_points = torch.from_numpy(eval_points).to("cuda",dtype=torch.float32)

    # prepare batch loss
    input_len_batch = torch.from_numpy(np.array(input_lengths * 2)).to("cuda", dtype=torch.int32)
    label_batch = label.repeat([2, 1])
    label_len_batch = torch.from_numpy(np.array([label.shape[1]] * 2)).to("cuda", dtype=torch.int32)

    _,loss_val = get_grad_batch(model, loss, eval_points, input_len_batch, label_batch, label_len_batch, model_name=model_name)
    print('lossl', loss_val)
    if model_name == 'wave2letter':
        input = input.unsqueeze(0)
        noise = np.expand_dims(noise, 1)
    # compute gradients by loss
    losses_tiled = np.tile(np.reshape(loss_val, (-1, 1)), np.prod(input.size()))
    losses_tiled = np.reshape(losses_tiled, (2,) + torch.squeeze(input,dim=0).shape)
    grad_val = np.mean(losses_tiled * noise, axis=0) / sigma
    grad_indices= np.mean(grad_val, axis=0, keepdims=True).reshape(-1)[indices]
    if np.max(grad_indices) == 0 and np.min(grad_indices) == 0:
        grad_indices = np.random.normal(size=grad_indices.shape)
    g_eval[indices] = grad_indices
    g_hat = g_eval.reshape(input.size())
    # g_hat = g_hat * 2
    # print(np.max(g_eval), np.min(g_eval))

    return np.mean(losses_tiled), g_hat, indices, prev_indice

def random_sample(input, k):
    cols = np.random.choice(input.shape[-1], k, replace=False)
    indice = []
    for i in cols:
        idx_col = np.linspace(i, 80 * input.shape[-1] + i, 81, dtype=int)
        indice.append(np.random.choice(idx_col, 1, replace=False))
    indice = np.concatenate(indice)
    return indice

def max_sample(input, k):
    cols = np.random.choice(input.shape[-1], k, replace=False)
    indice = []
    var = input.reshape(-1)
    for i in cols:
        idx_col = np.linspace(i, 80 * input.shape[-1] + i, 81, dtype=int)
        indice.append(idx_col[torch.argmin(var[idx_col])])

    # indice = np.concatenate(indice)
    return indice

def fd(model, ori_input, input, loss, label, input_lengths, batch_size, ori_loss, total_q, use_TD, model_name='ds2', prev_indice=[]):
    g_eval = np.zeros(shape=ori_input.size())
    # grad = np.zeros(shape=input.size()[-1] * input.size()[-2])
    indices = np.random.choice(input.size()[-1] * input.size()[-2], size=batch_size, replace=False)
    # indices = np.array(range(input.size()[-1] * input.size()[-2]))
    # indices = max_sample(ori_input, batch_size)
    # indices = random_sample(ori_input, batch_size)

    if use_TD==1:
        h = 3
        if total_q <= 3:  # initalize
            var_list = np.array(range(0, input.reshape(-1).size()[0]), dtype=np.int32)
            indices = var_list[np.random.choice(var_list.size, int(batch_size), replace=False)]  # selected coord index
        else:  # after the first iteration
            _, _, indice = td(ori_input, input, input.size()[-1], h, batch_size, prev_indice)
            prev_indice += indice.tolist()
            indices = np.array(indice)

    grad = np.zeros(shape=len(indices))

    # follow codes in ZOO: estimate in a batch
    var = np.repeat(np.zeros(shape=input.size()), 2, axis=0)
    var[0].reshape(-1)[indices] += 0.0001
    var[1].reshape(-1)[indices] -= 0.0001
    var = torch.tensor(var).to("cuda", dtype=torch.float32)
    var = var.reshape((-1,) + input.size()[1:])

    perturbed = input + var
    input_len_batch = torch.from_numpy(np.array(input_lengths * 2)).to("cuda", dtype=torch.int32)
    label_batch = label.repeat([2, 1])
    label_len_batch = torch.from_numpy(np.array([label.shape[1]] * 2)).to("cuda", dtype=torch.int32)

    _, perturbed_scores = get_grad_batch(model, loss, perturbed, input_len_batch, label_batch,
                                         label_len_batch, model_name=model_name)
    # perturbed_scores = loss(perturbed, input).cpu().detach().numpy()
    losses_tiled = np.tile(np.reshape(perturbed_scores, (-1, 1)), np.prod(input.size()))
    losses_tiled = np.reshape(losses_tiled, (2,) + torch.squeeze(input, dim=0).shape)

    for i in range(len(indices)):
        grad[i] = (losses_tiled[0].reshape(-1)[indices[i]] - losses_tiled[1].reshape(-1)[indices[i]]) / 0.0002
    if np.max(grad) == 0 and np.min(grad) == 0:
        grad = np.random.normal(size=grad.shape)
    g_eval.reshape(-1)[indices] = grad
    g_hat = g_eval.reshape(input.size())
    # g_hat = g_hat * 0.05
    # print(np.max(grad), np.min(grad))
    if model_name == 'ds2v1':
        g_hat = g_hat * 0.1

    return g_hat, indices, prev_indice



