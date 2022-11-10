import qiskit
import numpy as np
# Training hyperparameter
num_shots = 10000
learning_rate = 0.4
backend = qiskit.Aer.get_backend('aer_simulator')
lambdas = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]
# For parameter-shift rule
two_term_psr = {
    'r': 1/2,
    's': np.pi / 2
}

four_term_psr = {
    'alpha': np.pi / 2,
    'beta' : np.pi,
    'd_plus' : 1j,
    'd_minus': 1j*(1-np.sqrt(2))/2
}