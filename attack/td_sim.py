import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch import optim


def sim_loss(adv_input, h):
    total_loss = 0
    for s in range(0, adv_input.shape[1] - h):
        t1_b = adv_input[:, s:s + h]
        t2_b = adv_input[:, s + 1:s + h + 1]
        loss = nn.CosineEmbeddingLoss()(t1_b, t2_b, torch.tensor([-1]).cuda())
        total_loss += loss
    return total_loss / (adv_input.shape[1] - h)

def td_sim_2(adv_input, noise_samples=500, h=2):
    sim = []
    # adv_input = adv_input.detach().cpu().numpy()
    # adv_input = np.squeeze(adv_input)
    adv_input = torch.squeeze(adv_input)
    noise = torch.zeros(adv_input.shape).cuda()
    noise.requires_grad = True



    new_input = adv_input + noise

    loss = sim_loss(new_input, h=h)
    loss.backward(retain_graph=True)

    grad = noise.grad.squeeze() * (-1)
    grad = grad.reshape(1, -1)
    grad_sort = torch.flip(torch.argsort(grad), [0])
    perturb_indice_1 = grad_sort[0, :noise_samples//2]
    perturb_indice_2 = grad_sort[0, -noise_samples//2:]
    perturb_indice = torch.cat([perturb_indice_1, perturb_indice_2])

    # work_perturb = torch.zeros_like(adv_input).reshape(1, -1)
    # work_perturb[0, perturb_indice_1] = torch.sign(grad[0, perturb_indice_1]) * 0.5
    # work_perturb[0, perturb_indice_2] = torch.sign(grad[0, perturb_indice_2]) * 0.5
    # work_perturb = work_perturb.reshape(adv_input.shape)
    # work_perturb = work_perturb.unsqueeze(0)
    # work_perturb = work_perturb.unsqueeze(0)
    # print(torch.max(work_perturb), torch.min(work_perturb))
    # return work_perturb.detach().cpu().numpy()
    return perturb_indice


def td_sim(ori_input, adv_input, h=3, batch_size=200, change_list=[], select='best'):
    sim = []
    adv_input = adv_input.detach().cpu().numpy()
    adv_input = np.squeeze(adv_input)
    for s in range(0, adv_input.shape[1] - 3):
        t1_b = adv_input[:, s:s + h]
        t2_b = adv_input[:, s+1:s + h + 1]
        cs_ae = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        sim.append(cs_ae[0][0])

    if select == 'best':
        selected_col = 0
        ind_max = np.argsort(np.array(sim))[::-1]
        for i in ind_max:
            if i not in change_list:
                selected_col = i
                change_list.append(i)
                break
    elif select == 'random':
        selected_col = np.random.randint(0, len(sim))
    target_t1 = adv_input[:, selected_col:selected_col+h].reshape(1, -1)
    target_t2 = adv_input[:, selected_col+1:selected_col+h+1].reshape(1, -1)
    tensor_t1 = torch.from_numpy(target_t1)
    tensor_t2 = torch.from_numpy(target_t2)
    noise = torch.zeros(tensor_t1.shape)
    noise.requires_grad = True
    optimizer = optim.Adam(params=[noise], lr=0.2)

    for i in range(5):
        new_t1 = tensor_t1 + noise
        optimizer.zero_grad()
        loss = nn.CosineEmbeddingLoss()(new_t1, tensor_t2, torch.tensor([-1]))
    # loss = sim_loss(adv_input, h)
        loss.backward(retain_graph=True)
        # noise = noise - noise.grad
        optimizer.step()
    # print(sim_loss)

    # noise = noise - noise.grad
    # optimizer.step()

    # new_t1 = tensor_t1 - torch.sign(noise.grad) * 0.5
    # sim_loss = nn.CosineEmbeddingLoss()(new_t1, tensor_t2, torch.tensor([-1]))
    # print(sim_loss)

    grad = noise.squeeze()
    grad_sort = torch.flip(torch.argsort(grad), [0])
    perturb_indice_1 = grad_sort[:batch_size//2]
    perturb_indice_2 = grad_sort[-batch_size//2:]
    work_perturb = torch.zeros_like(tensor_t1)
    # work_perturb[0, perturb_indice_1] = grad[perturb_indice_1]
    work_perturb[0, perturb_indice_1] = grad[perturb_indice_1]
    # work_perturb[0, perturb_indice_2] = grad[perturb_indice_2]
    work_perturb[0, perturb_indice_2] = grad[perturb_indice_2]

    # tensor_t1[0, perturb_indice] += noise[0, perturb_indice]
    # tensor_t1 += noise
    # tensor_t1 = tensor_t1.reshape(-1, h)

    perturb = torch.zeros(1, 1, adv_input.shape[0], adv_input.shape[1])
    perturb[0, 0, :, selected_col:selected_col+h] = work_perturb.reshape(-1, h)
    # adv_input[:, ] = tensor_t1
    # print(perturb_indice_1)
    print(selected_col, torch.max(work_perturb), torch.min(work_perturb))
    return perturb.detach().numpy(), change_list
    # print(nn.CosineSimilarity()(tensor_t1, tensor_t2))


def get_tdsim_indice(adv_input, h=3, batch_size=200, select='best', change_list=[]):
    sim = []
    adv_input = adv_input.detach().cpu().numpy()
    adv_input = np.squeeze(adv_input)
    for s in range(0, adv_input.shape[1] - h):
        t1_b = adv_input[:, s:s + h]
        t2_b = adv_input[:, s+1:s + h + 1]
        cs_ae = cosine_similarity([t1_b.ravel().tolist()], [t2_b.ravel().tolist()])
        sim.append(cs_ae[0][0])

    if select == 'best':
        ind_max = np.argsort(np.array(sim))[::-1]
        for i in ind_max:
            if i not in change_list:
                selected_col = i
                change_list.append(i)
                if len(change_list) == len(adv_input[1]):
                    change_list = []    
                break
        # selected_col = ind_max[0]
    elif select == 'random':
        selected_col = np.random.randint(0, len(sim))

    target_t1 = adv_input[:, selected_col:selected_col+h].reshape(1, -1)
    target_t2 = adv_input[:, selected_col+1:selected_col+h+1].reshape(1, -1)
    tensor_t1 = torch.from_numpy(target_t1)
    tensor_t2 = torch.from_numpy(target_t2)
    noise = torch.zeros(tensor_t1.shape)
    noise.requires_grad = True
    optimizer = optim.Adam(params=[noise], lr=0.1)

    for i in range(5):
        new_t1 = tensor_t1 + noise
        optimizer.zero_grad()
        loss = nn.CosineEmbeddingLoss()(new_t1, tensor_t2, torch.tensor([-1]))
    # loss = sim_loss(adv_input, h)
        loss.backward(retain_graph=True)
        # noise = noise - noise.grad
        optimizer.step()

    grad = noise.squeeze()
    grad_sort = torch.flip(torch.argsort(grad), [0])
    perturb_indice_1 = grad_sort[:batch_size//2]
    perturb_indice_2 = grad_sort[-batch_size//2:]

    perturb_indice = torch.cat([perturb_indice_1, perturb_indice_2])
    return perturb_indice.numpy(), change_list