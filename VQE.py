import tqix, constant, base, lagrange_psr
import numpy as np

n = 5
lambdax = 0.05

def H_LMG(h, lambdax, n, Jx, Jy, Jz):
    return -2*h*Jz - 2*lambdax/n*(Jx**2 - Jy**2)

def cost_function_noise(thetas, h):
    qc = tqix.circuit(n)
    for i in range(0, n):
        qc.RX(thetas[i], i)
        qc.RZ(thetas[i + n], i)
        qc.RX(thetas[i + 2 * n], i)
    Jx = qc.Jx()
    Jy = qc.Jy()
    Jz = qc.Jz()
    h_LMG = H_LMG(h, lambdax, n, Jx, Jy, Jz)
    return np.real(np.trace((h_LMG @ qc.state).toarray()))

def cost_function(thetas, h):

    qc = tqix.circuit(n)
    for i in range(0, n):
        qc.RX(thetas[i])
        qc.RZ(thetas[i + n])
        qc.RX(thetas[i + 2 * n])
    Jx = qc.Jx()
    Jy = qc.Jy()
    Jz = qc.Jz()
    h_LMG = H_LMG(h, lambdax, n, Jx, Jy, Jz)
    return np.real(np.trace((h_LMG @ qc.state).toarray()))

def optimal(h):
    costs = []
    thetass = []
    thetas = np.random.uniform(0, 2*np.pi, n*3)
    for i in range(0, 30):
        print("Iteration: ", i)
        thetass.append(thetas)
        thetas = thetas - constant.learning_rate*base.two_prx_hLMG(cost_function, thetas, h)
        costs.append(cost_function(thetas, h))
    np.savetxt("cost_" + str(h) + ".txt", costs)
    np.savetxt("thetas_" + str(h) + ".txt", thetass)

def optimal_noise(h):
    costs = []
    thetass = []
    thetas = np.random.uniform(0, 2*np.pi, n*3)
    for i in range(0, 10):
        print("Iteration: ", i)
        thetass.append(thetas)
        print(thetas)

        thetas = thetas - constant.learning_rate*base.two_prx_hLMG(cost_function_noise, thetas, h)
        costs.append(cost_function(thetas, h))
        print(costs)
    np.savetxt("cost_" + str(h) + ".txt", costs)
    np.savetxt("thetas_" + str(h) + ".txt", thetass)
