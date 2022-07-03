from tqix import *
from tqix.pis import *
import numpy as np
N = 20 #qubits
qc = circuit(N)
qc.RN(-np.pi/2,np.pi/4)
prob = qc.measure(num_shots=1000)
#to get state information
psi = qc.state #sparse matrix
psi = qc.state.toarray() #full matrix
print(psi)