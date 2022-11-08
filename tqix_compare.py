
import numpy as np
import VQE
hs = np.round(np.arange(0, 0.1006, 0.006), 3)
costs = []
for h in hs:
    print(h)
    VQE.optimal(h)
