"""
Unit tests for cim_svp.extras.prune
"""

import numpy as np
from cim_svp.extras.prune import prune_basis


def test_std_filter():
    B = np.eye(4, dtype=object)
    # add a long outlier
    B = np.vstack([B, 100 * np.ones(4, dtype=object)])
    pruned, stats = prune_basis(B, norm_mode="std", norm_param=1.5,
                                dir_threshold_deg=1.0e-3)  # tiny threshold so only norm matters
    assert len(pruned) == 4           # outlier removed
    assert stats["removed_norm"] == 1


def test_direction_filter():
    v1 = np.array([1, 0, 0], dtype=object)
    v2 = np.array([2, 0, 0], dtype=object)   # same direction
    v3 = np.array([0, 1, 0], dtype=object)   # orthogonal
    pruned, stats = prune_basis([v1, v2, v3],
                                norm_mode=None,
                                dir_threshold_deg=5.0)
    assert len(pruned) == 2
    assert stats["removed_dir"] == 1


def test_log_factor():
    v_small = np.array([1, 0, 0], dtype=object)
    v_big   = np.array([1000, 0, 0], dtype=object)
    pruned, _ = prune_basis([v_small, v_big],
                            norm_mode="log_factor", norm_param=0.3,
                            dir_threshold_deg=1.0e-3)
    # factor 0.3 ≈ ±1/2 order of magnitude, should keep none
    assert len(pruned) == 0