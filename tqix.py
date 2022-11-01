from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import torch
import numpy as np

n = 5
loss_dict = {}
lambdax = 0.05
hs = [-0.1, 0, 0.1]

def H_LMG(h, lambdax, n, Jx, Jy, Jz):
    return -2*h*Jz - 2*lambdax/n*(Jx**2 - Jy**2)


def cost_function(thetas):
    qc = circuit(n)
    for i in range(0, n, 3):
        qc.RX(thetas[i], 0)
        qc.RZ(thetas[i + 1], 1)
        qc.RX(thetas[i + 2], 2)

    Jx = qc.Jx()
    Jy = qc.Jy()
    Jz = qc.Jz()

    h_LMG = H_LMG(hs[0], lambdax, n, Jx, Jy, Jz)
    psi = qc.state
    return np.real(np.trace((h_LMG @ psi).toarray()))



# function to optimize circuit of sparse array


def sparse(optimizer, loss_dict, mode):
    def objective_function(params): return cost_function(params)
    init_params = [2.9644759 , 5.17531829, 2.70746915, 0.34326549, 4.22412149,
       5.59371018, 4.84429238, 1.80126693, 4.56341304, 5.23205335,
       5.8469486 , 5.50293912, 4.16056373, 2.72112783, 3.12698443]  # random init parameters
    _, _, _, loss_hist, time_iters = fit(
        objective_function, optimizer, init_params,
        return_loss_hist=True, return_time_iters=True)
    loss_dict[mode] = loss_hist
    return loss_dict, time_iters
# function to optimize circuit of tensor


def tensor(optimizer, loss_dict, mode):
    objective_function = lambda params: cost_function(params, use_gpu=True)
    init_params = [2.9644759 , 5.17531829, 2.70746915, 0.34326549, 4.22412149,
       5.59371018, 4.84429238, 1.80126693, 4.56341304, 5.23205335,
       5.8469486 , 5.50293912, 4.16056373, 2.72112783, 3.12698443]  # random init parameters
    init_params = torch.tensor(init_params).to('cuda').requires_grad_()

    _, _, _, loss_hist, time_iters = fit(
        objective_function, optimizer, init_params,
        return_loss_hist=True, return_time_iters=True)
    loss_dict[mode] = loss_hist
    return loss_dict, time_iters


optimizer = GD(lr=0.0001, eps=1e-10, maxiter=200, tol=1e-19, N=n)
loss_dict, _ = tensor(optimizer, loss_dict, "tensor_gd")
