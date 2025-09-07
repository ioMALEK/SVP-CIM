import numpy as np, pytest
from cim_svp.extras import classical_solvers as cs

B2 = np.array([[2,1],
               [1,1]], dtype=object)           # det = 1, shortest = (1,-1)

def test_enum_dim2():
    v, n = cs.shortest_enum(B2)
    assert n == pytest.approx((2)**0.5)
    assert tuple(v) in {(1,-1), (-1,1)}

def test_mc_sampling():
    v, n = cs.shortest_monte_carlo(B2, samples=1000, seed=0)
    assert n >= (2)**0.5          # never shorter than true optimum

def test_lll_length_bound():
    B = np.random.randint(-5, 6, size=(6, 6)).astype(object)
    Bred = cs.lll_reduce(B)
    assert np.linalg.det(B.astype(float)) != 0
    l2 = np.linalg.norm(Bred[0].astype(float))
    bound = 2**(5/2) * abs(np.linalg.det(B.astype(float)))**(1/6)
    assert l2 <= bound * 1.1       # slack for float rounding