import numpy as np
import torch


def get_grad(model, loss_f, spec, sepc_len, label):
    # spec.requires_grad = True
    y_hat = model(spec,torch.IntTensor(sepc_len).cuda())
    output = y_hat.transpose(1,0)
    l = loss_f(output,label,sepc_len,[label.shape[1]])
    # model.zero_grad()
    # l.backward()
    return y_hat, l

def get_grad_q(model, loss_f, spec, sepc_len, label, label_length):
    y_hat,output_sizes = model(spec,sepc_len)
    out = y_hat.transpose(0, 1)
    out = out.log_softmax(-1)
    l = loss_f(out, label, output_sizes, label_length)
    return y_hat, l


def nes(model, input, loss, label, input_lengths, batch_size, sigma):
    """https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/NES.py"""
    g_eval = np.zeros(input.size()).reshape(-1)
    var_len = input.reshape(-1).size()[0]
    var_list = np.array(range(0, var_len), dtype=np.int32)
    indices = var_list[np.random.choice(var_list.size, int(100), replace=False)]
    indices = np.array(indices)

    ## gradient esitmation by nes
    noise_pos = np.random.normal(size=input.size())
    noise = np.concatenate([noise_pos, -noise_pos], axis=0)
    eval_points = input.cpu().detach().numpy() + sigma * noise
    eval_points = torch.from_numpy(eval_points).to("cuda",dtype=torch.float32)

    # prepare batch loss
    input_len_batch = torch.from_numpy(np.array(input_lengths * 2)).to("cuda", dtype=torch.int32)
    label_batch = label.repeat([2, 1])
    label_len_batch = torch.from_numpy(np.array([label.shape[1]] * 2)).to("cuda", dtype=torch.int32)

    _,loss_val = get_grad_q(model, loss, eval_points, input_len_batch, label_batch, label_len_batch)
    losses_tiled = np.tile(np.reshape(loss_val.cpu().detach().numpy(), (-1, 1)), np.prod(input.size()))
    losses_tiled = np.reshape(losses_tiled, (2,) + torch.squeeze(input,dim=0).shape)
    grad_val = np.mean(losses_tiled * noise, axis=0) / sigma
    grad_indices= np.mean(grad_val, axis=0, keepdims=True).reshape(-1)[indices]
    g_eval[indices] = grad_indices

    return np.mean(losses_tiled), g_eval.reshape(input.size())


def stochastic_coord(model, input, loss, label,input_lengths, batch_size, sigma, indice):
    """Implementation of ZOO algorithm1: stochastic coordinate descent.
        Add noise to coordinates indices. """

    # random sampling s coords among input: only update these coords
    g_eval = np.zeros(input.shape)
    # evaluation on these coordinates by randomly adding noise to these selected coords

    ui = np.zeros(shape=(batch_size,) + torch.squeeze(input,dim=0).shape)
    for i in range(batch_size):
        ui_i_flatten = ui[i].ravel()
        ui_i_flatten[indice[i]] = np.random.normal(size=1)
        ui[i] = np.reshape(ui_i_flatten, input.shape)

    input_batch = np.tile(input.cpu().detach().numpy(),(batch_size,1,1,1))
    eval_points_up,eval_points_down = input_batch + ui, input_batch - ui
    eval_points_up = torch.from_numpy(eval_points_up).to("cuda", dtype=torch.float)
    eval_points_down = torch.from_numpy(eval_points_down).to("cuda", dtype=torch.float)

    # estimate grandients
    input_len_batch = torch.from_numpy(np.array(input_lengths * batch_size)).to("cuda", dtype=torch.int32)
    label_batch = label.repeat([batch_size, 1])
    label_len_batch = torch.from_numpy(np.array([label.shape[1]] * batch_size)).to("cuda", dtype=torch.int32)

    y_hat_up, l_up = get_grad_q(model, loss, eval_points_up, input_len_batch, label_batch, label_len_batch)
    y_hat_down, l_down= get_grad_q(model, loss, eval_points_down, input_len_batch, label_batch, label_len_batch)
    g_eval = g_eval + (l_up.item() - l_down.item()) / (2*sigma)
    g_eval = np.mean(g_eval, axis=0, keepdims=True)
    return g_eval


def zoo_adam(model, input, loss, label,input_lengths, indice, sigma, mt_arr, vt_arr, adam_epoch, real_modifier, step_size,beta1, beta2):
    """Implmentation of ZOO_ADAM (ZOO algorithm2)"""
    # print("real m before adam:", real_modifier)
    ## get modifer from inital
    #get gradient from (6): two-side nes
    g_eval = np.zeros(input.shape)
    # eval_points = input.cpu().detach().numpy() + sigma * ((1-B)*ui[i] + B*prior)
    ui = real_modifier.reshape(input.shape)
    eval_points_up = input.cpu().detach().numpy() + sigma * ui
    eval_points_down = input.cpu().detach().numpy() - sigma * ui
    eval_points_up = torch.from_numpy(eval_points_up).to("cuda", dtype=torch.float)
    eval_points_down = torch.from_numpy(eval_points_down).to("cuda", dtype=torch.float)
    _, loss_eval_up = get_grad(model, loss, eval_points_up, input_lengths, label)
    _, loss_eval_down = get_grad(model, loss, eval_points_down, input_lengths, label)
    g_eval = g_eval + (loss_eval_up.item() - loss_eval_down.item())/ (2*sigma)* ui

    # get corresponded gradient of selected coords
    g_adam = g_eval.reshape(-1)[indice]

    # update moving average by beta1 and beta2
    mt = mt_arr[indice] #select coord
    mt = beta1 * mt + (1 - beta1) * g_adam
    mt = mt/(1-beta1)
    mt_arr[indice] = mt

    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (g_adam * g_adam)
    vt = vt/(1-beta2)
    vt_arr[indice] = vt

    # update parameters
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    old_val = real_modifier[indice]
    old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-4)
    real_modifier[indice] = old_val
    adam_epoch[indice] = epoch + 1

    # update gradients
    g_fin = g_eval + real_modifier.reshape(input.size())
    # print("real m after adam:", real_modifier)
    return g_fin


def zoo_adam(model, input, loss, label,input_lengths, indice, sigma, mt_arr, vt_arr, adam_epoch, real_modifier, step_size,beta1, beta2):
    """Implmentation of ZOO_ADAM (ZOO algorithm2)"""
    # print("real m before adam:", real_modifier)
    ## get modifer from inital
    #get gradient from (6): two-side nes
    g_eval = np.zeros(input.shape)
    # eval_points = input.cpu().detach().numpy() + sigma * ((1-B)*ui[i] + B*prior)
    ui = real_modifier.reshape(input.shape)
    eval_points_up = input.cpu().detach().numpy() + sigma * ui
    eval_points_down = input.cpu().detach().numpy() - sigma * ui
    eval_points_up = torch.from_numpy(eval_points_up).to("cuda", dtype=torch.float)
    eval_points_down = torch.from_numpy(eval_points_down).to("cuda", dtype=torch.float)
    _, loss_eval_up = get_grad(model, loss, eval_points_up, input_lengths, label)
    _, loss_eval_down = get_grad(model, loss, eval_points_down, input_lengths, label)
    g_eval = g_eval + (loss_eval_up.item() - loss_eval_down.item())/ (2*sigma)* ui

    # get corresponded gradient of selected coords
    g_adam = g_eval.reshape(-1)[indice]

    # update moving average by beta1 and beta2
    mt = mt_arr[indice] #select coord
    mt = beta1 * mt + (1 - beta1) * g_adam
    mt = mt/(1-beta1)
    mt_arr[indice] = mt

    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (g_adam * g_adam)
    vt = vt/(1-beta2)
    vt_arr[indice] = vt

    # update parameters
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    old_val = real_modifier[indice]
    old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-4)
    real_modifier[indice] = old_val
    adam_epoch[indice] = epoch + 1

    # update gradients
    g_fin = g_eval + real_modifier.reshape(input.size())
    # print("real m after adam:", real_modifier)
    return g_fin


def zoo_newton(model, input, loss, label,input_lengths, real_modifier, sigma, spec_loss,indice):
    # https://github.com/as791/ZOO_Attack_PyTorch/blob/ab003a1c16944a82b6ab9bfb392cebd8c0dc7814/zoo_l2_attack_black.py#L42
    lr = 0.0005
    g_eval_1 = np.zeros(input.shape)
    g_eval_2 = np.zeros(input.shape)
    ui = real_modifier.reshape(input.shape)
    #for i in range(s):
    #ui[i] = ui[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(ui[i]))))
    # eval_points = input.cpu().detach().numpy() + sigma * ((1-B)*ui[i] + B*prior)
    eval_points_up = input.cpu().detach().numpy() + sigma * ui
    eval_points_down = input.cpu().detach().numpy() - sigma * ui
    eval_points_up = torch.from_numpy(eval_points_up).to("cuda", dtype=torch.float)
    eval_points_down = torch.from_numpy(eval_points_down).to("cuda", dtype=torch.float)
    _, loss_eval_up = get_grad(model, loss, eval_points_up, input_lengths, label)
    _, loss_eval_down = get_grad(model, loss, eval_points_down, input_lengths, label)
    g_eval_1 = (loss_eval_up.item() - loss_eval_down.item()) * ui/ (2 * sigma) # est order 1
    g_eval_2 = (loss_eval_up.item() - 2*spec_loss.item() + loss_eval_down.item()) * ui/ (sigma*sigma)

    # https://github.com/IBM/ZOO-Attack/blob/master/l2_attack_black.py
    # negative hessian cannot provide second order information, just do a gradient descent
    g_eval_2[g_eval_2<0] = 1.0
    # hessian too small, could be numerical problems
    g_eval_2[g_eval_2 < 0.1] = 0.1

    # if g_eval_2 <= 0:
    #     g_hat = (-1) * lr * g_eval_1
    # else:
    #     g_hat = (-1) * lr * (g_eval_1/g_eval_2)

    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * g_eval_1.reshape(-1)[indice] / g_eval_2.reshape(-1)[indice]
    # set it back to [-0.5, +0.5] region

    m[indice] = old_val
    # print(m[indice])
    g_eval = real_modifier.reshape(input.size())
    return g_eval

def zoo_pca(model, input, loss, label,input_lengths,sigma, k):
    """Implementation of ZOO PCA. Different from the above methods,
    ZOOPCA uses principle components (top-k eigenvectors) as noise to compute losses.
    Steps follow: https://github.com/sunblaze-ucb/blackbox-attacks/blob/5ee7bb8693aec0a0c7f3590b0e3c7686986a7eb3/query_based_attack.py
    Inputs:
    k(int): the number of principle components
    """

    ## method based on the paper
    # use PCA to find right direction
    input_pca = torch.squeeze(input).cpu().detach().numpy()
    _, U = PCA(input_pca, k) #each row is an eigenvector

    # get estimated losses and gradients
    g_pca = np.zeros(input.shape)
    for i in range(k):
        # get loss
        ui = U[i]/np.linalg.norm(U[i],ord=2)
        eval_points_up = input.cpu().detach().numpy() + sigma * ui
        eval_points_down = input.cpu().detach().numpy() - sigma * ui
        eval_points_up = torch.from_numpy(eval_points_up).to("cuda", dtype=torch.float)
        eval_points_down = torch.from_numpy(eval_points_down).to("cuda", dtype=torch.float)
        _, loss_eval_up = get_grad(model, loss, eval_points_up, input_lengths, label)
        _, loss_eval_down = get_grad(model, loss, eval_points_down, input_lengths, label)
        g_eval = (loss_eval_up.item() - loss_eval_down.item()) / (2*sigma)
        g_pca = g_pca + g_eval*ui*ui

    return g_pca