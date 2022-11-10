import numpy as np
import base
from scipy.linalg import null_space

def solve_linear_equation(T):
    """Solve the equation T @ d = 0, d <> 0.

    Args:
        T (numpy.ndarray): list of T matrix
    """
    init_rcond = 1
    t = 0
    while t != 1000:
        d = (null_space(T, rcond = init_rcond))
        t += 1
        if d.shape[1] != 1:
            init_rcond /= 10
        else:
            return d
    return []
def lagrange_psr(lambdas):
    """Find scalar factors and shift values
    After that, conduct lagrange parameter-shift rule
    Args:
        lambdas (list): list of eigenvalues

    Returns:
        [numpy.ndarray, numpy.ndarray]: Scalar factors and Shift values
    """
    dim_d = int(len(lambdas)**2/4) - 1 
    # Find T
    while(True): 
        Ts = []
        deltas = []
        thetas = []
        for i in range(0, dim_d):
            theta = np.random.uniform(0, 2*np.pi)
            delta = (base.calculate_Lambda_matrix(lambdas, theta) - base.calculate_Lambda_matrix(lambdas, -theta))
            thetas.append(theta)
            deltas.append(delta)
            Ts.append(base.upper_matrix(delta))
        T = Ts[0]
        for i in range(1, len(Ts)):
            T = np.hstack((T, Ts[i]))

        d = solve_linear_equation(T)
        if len(d) > 0:
            break

    for i in range(0, len(Ts)):
        if np.abs((T @ d)[0])[0] > 10**(-10):
            return False
    
    # Get normalized scalar
    sumMatrix = d[0] * deltas[0]
    for i in range(1, len(Ts)):
        sumMatrix += d[i] * deltas[i]
    # sumMatrix = (np.round(sumMatrix, 3))

    # Normalize d
    d /= sumMatrix[0][1]
    return thetas, d