"""
Unit tests for cim_svp.extras.scale
"""

import numpy as np
import pytest
from cim_svp.extras.scale import scale_candidates


def test_duplicate_removal_floor():
    v1 = np.array([1, 0], dtype=object)
    v2 = np.array([2, 0], dtype=object)     # same direction, larger norm
    scaled, stats = scale_candidates([v1, v2], rounding="floor")
    # After scaling: v1 → [2,0]; v2 → [2,0]  → duplicates collapse to one
    assert len(scaled) == 1
    assert stats["duplicates_removed"] == 1


def test_duplicate_removal_ceil():
    v1 = np.array([1, 0], dtype=object)
    v2 = np.array([2, 0], dtype=object)
    scaled, _ = scale_candidates([v1, v2], rounding="ceil")
    # Ceil makes v2 scale to itself, v1→[2,0] as well → duplicates collapse
    assert len(scaled) == 1


def test_cap_and_stats():
    v1 = np.array([1, 0, 0], dtype=object)
    v2 = np.array([3, 0, 0], dtype=object)
    scaled, stats = scale_candidates([v1, v2], k_cap=2)
    assert stats["max_scale"] == 2
    # scaling factor of v2 should be 1 (already the max norm)
    assert stats["min_scale"] == 1


def test_bad_rounding():
    v = np.array([1, 0], dtype=object)
    with pytest.raises(ValueError):
        scale_candidates([v], rounding="avg")