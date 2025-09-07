# cim_optimizer/simcpm.py  (overwrite current stub)

import numpy as np, torch
from cim_optimizer.CAC_Potts import CIM_CAC_Potts_GPU

class CPM:
    def __init__(self, J, Q=3, **kw):
        self.J = J.astype(np.float64)
        self.Q = Q
        self.kw = kw
        self._spins = None
    def run(self):
        spins, *_ = CIM_CAC_Potts_GPU(J=self.J, Q=self.Q, **self.kw)
        self._spins = spins
    def spins(self):  return self._spins
    def energy(self):
        s = self._spins
        same = (s[:, :, None] == s[:, None, :])
        return -0.5 * np.sum(self.J * same, axis=(1,2))