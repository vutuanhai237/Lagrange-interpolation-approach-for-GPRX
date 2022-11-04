import tqix, constant, base
import numpy as np
import importlib
importlib.reload(constant)
importlib.reload(base)
n = 5
lambdax = 0.05

def H_LMG(h, lambdax, n, Jx, Jy, Jz):
    return -2*h*Jz - 2*lambdax/n*(Jx**2 - Jy**2)

def cost_function(thetas, h):

    qc = tqix.circuit(n)
    for i in range(0, n):
        qc.RX(thetas[i], i)
        qc.RZ(thetas[i + n], i)
        qc.RX(thetas[i + 2 * n], i)
        
    Jx = qc.Jx()
    Jy = qc.Jy()
    Jz = qc.Jz()
    h_LMG = H_LMG(h, lambdax, n, Jx, Jy, Jz)
    psi = qc.state
    
    return np.real(np.trace((h_LMG @ psi).toarray()))

def optimal(h):
    costs = []
    thetass = []
    thetas = np.ones(n*3)
    for i in range(0, 20):
        thetass.append(thetas)
        thetas = thetas - constant.learning_rate*base.two_prx(cost_function, thetas, h)
        costs.append(cost_function(thetas, h))
    np.savetxt("cost_" + str(h) + ".txt", costs)
    np.savetxt("thetas_" + str(h) + ".txt", thetass)

hs = np.round(np.arange(-0.1, 0.1, 0.002), 3)
for h in hs:
    print(h)
    optimal(h)