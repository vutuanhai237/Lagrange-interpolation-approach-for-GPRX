import qiskit
import numpy as np
# Training hyperparameter
num_shots = 10000
backend = qiskit.Aer.get_backend('qasm_simulator')

# For parameter-shift rule
two_term_psr = {
    'r': 1/2,
    's': np.pi / 2
}

four_term_psr = {
    'alpha': np.pi / 2,
    'beta' : 3 * np.pi / 2,
    'd_plus' : (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}