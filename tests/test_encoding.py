"""
Fast tests for the upgraded build_extended_subspace
"""

import numpy as np
import pytest
from cim_svp.extras.encoding import build_extended_subspace


def test_range_support_and_coeffs():
    B = np.eye(3, dtype=object)
    stats, cand = build_extended_subspace(
        B,
        coeff_min=1, coeff_max=2, coeff_interval=1,
        support_min=1, support_max=2
    )
    assert stats["n_vectors"] == len(cand)
    assert stats["duplicates_removed"] == 0
    # norms should be 1,2, sqrt(2), sqrt(5) ... test the min
    assert min(vn for _, vn in cand) == 1


def test_explicit_coeff_and_all_support():
    B = np.eye(2, dtype=object)
    stats, cand = build_extended_subspace(
        B, coeff_values=[3], support_min="all"
    )
    # With coeff 3 and 2-dim basis we expect
    # ±3e1 , ±3e2 , ±3e1 ±3e2  => 8 vectors
    assert len(cand) == 8
    assert stats["n_vectors"] == 8


def test_skip_basis_rows():
    # row [1,0] should be filtered out
    B = np.array([[1, 0], [0, 1]], dtype=object)
    _, cand = build_extended_subspace(
        B, coeff_values=[1], support_min=1, support_max=1
    )
    vectors = [tuple(int(x) for x in v) for v, _ in cand]
    assert (1, 0) not in vectors and (0, 1) not in vectors

def test_bad_parameters():
    B = np.eye(2, dtype=object)
    with pytest.raises(ValueError):
        build_extended_subspace(B, coeff_min=2, coeff_max=1)  # min>max