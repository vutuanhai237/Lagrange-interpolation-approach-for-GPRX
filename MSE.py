import base, constant
import numpy as np, qiskit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import importlib
importlib.reload(constant)
importlib.reload(base)
step_sizes = base.create_log_step_sizes(0.001, 3, 0.1)
thetas = np.asarray([np.pi/2, np.pi/3, np.pi/6])
def u(qc, thetas):
    qc.rx(thetas[0], 0)
    qc.rz(thetas[1], 1)
    qc.cry(thetas[2], 0, 1)
    return qc
def f(thetas):
    qc = qiskit.QuantumCircuit(2,2)
    qc = u(qc, thetas)
    return base.measure(qc, [0, 1])

true_grad = base.true_grad(thetas)

mean_grad_stds = []
mean_grad_finites = []
for step_size in step_sizes:
    print(step_size)
    grad_finites = []
    grad_stds = []

    for _ in range(0, 1000):
        grad_finite = []
        grad_std = []
        for j in range(0, thetas.shape[0]):
            # if j != 2:
            #     grad_std.append(base.pseudo_two_prx(f, thetas, j, step_size))
            #     # grad_finite.append(base.two_finite_diff(f, thetas_origin, j, step_size))
            # else:
            grad_std.append(base.two_finite_diff(f, thetas, j, step_size))
            grad_finite.append(base.two_finite_diff(f, thetas, j, step_size))
        
        grad_finites.append(grad_finite)
        grad_stds.append(grad_std)

    mean_grad_stds.append(np.mean(grad_stds,axis = 0))
    mean_grad_finites.append(np.mean(grad_finites,axis = 0))
np.savetxt('mean_grad_stds.txt', mean_grad_stds)
np.savetxt('mean_grad_finites.txt', mean_grad_finites)