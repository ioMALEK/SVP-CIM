import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# 3D toy basis (random or fixed for demonstration)
basis = np.array([[2, 1, 0],
                  [0, 1, 1],
                  [1, 0, 2]])

coeff_vals = [-2, -1, 0, 1, 2]

# All possible Potts-encoded lattice vectors (multi-level coefficients)
potts_norms = []
potts_cands = []
for c in product(coeff_vals, repeat=3):
    if c == (0,0,0):
        continue
    v = sum(c[i]*basis[i] for i in range(3))
    norm = np.linalg.norm(v)
    potts_norms.append(norm)
    potts_cands.append((norm, c))

# All possible Ising-only combinations (±1 only)
ising_norms = []
ising_cands = []
for c in product([-1, 1], repeat=3):
    v = sum(c[i]*basis[i] for i in range(3))
    norm = np.linalg.norm(v)
    ising_norms.append(norm)
    ising_cands.append((norm, c))

# Find minimums
min_potts = min(potts_norms)
min_ising = min(ising_norms)

plt.figure()
plt.hist(potts_norms, bins=40, alpha=0.6, label='Potts (all integer coeffs)')
plt.hist(ising_norms, bins=20, alpha=0.6, label='Ising (±1 only)')
plt.axvline(min_potts, color='b', linestyle='--', label=f'Potts min: {min_potts:.2f}')
plt.axvline(min_ising, color='orange', linestyle='--', label=f'Ising min: {min_ising:.2f}')
plt.xlabel('Norm of candidate vector')
plt.ylabel('Frequency')
plt.title('Potts vs. Ising: Norm distribution of lattice vectors (toy 3D)')
plt.legend()
plt.tight_layout()
plt.savefig('potts_vs_ising_toy_histogram.pdf')
plt.show()
