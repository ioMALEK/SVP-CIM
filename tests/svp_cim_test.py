"""
tests/test_cim_svp.py
Fast sanity-check for the cim_svp helper package.
Run with:  pytest -q
"""

from pathlib import Path
import io, sys, types, numpy as np, pytest
from cim_svp import (
    load_lattice, save_yaml,
    vec_sq_norm_int, gram_int, build_J, sq_from_spins, sqrt_int, to_pm1,
    extract_spins, silence_stdio,
    make_seed, GLOBAL_SEED,
    tqdm_joblib
)
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# ------------------------------------------------------------------ #
#  IO module                                                         #
# ------------------------------------------------------------------ #
def test_load_lattice_literal(tmp_path: Path):
    basis_txt = "[[1,2],[3,4]]"
    file_ = tmp_path / "dim2_seed0.txt"
    file_.write_text(basis_txt)
    B = load_lattice(2, 0, folder=tmp_path)
    assert B.shape == (2, 2)
    assert B.dtype == object


def test_load_lattice_square(tmp_path: Path):
    rows = [" ".join(map(str, range(1, 6))) for _ in range(5)]
    txt = "\n".join(rows)
    (tmp_path / "dim5_seed0.txt").write_text("[" + txt + "]")
    B = load_lattice(5, 0, folder=tmp_path)
    assert B.shape == (5, 5)


def test_save_yaml(tmp_path: Path):
    d = dict(a=1, b=[2, 3])
    p = tmp_path / "out.yml"
    save_yaml(d, p)
    assert p.is_file() and "a: 1" in p.read_text()


# ------------------------------------------------------------------ #
#  Maths module                                                      #
# ------------------------------------------------------------------ #
def test_math_helpers():
    v = np.array([3, 4], dtype=object)
    assert vec_sq_norm_int(v) == 25
    B = np.array([[1, 0], [0, 1]], dtype=object)
    G = gram_int(B)
    np.testing.assert_array_equal(G, np.eye(2, dtype=object))
    J = build_J(B)
    expected = -np.eye(2)
    np.fill_diagonal(expected, 0.0)   # build_J clears the diagonal
    assert np.allclose(J, expected)
    s = np.array([1, -1])
    assert sq_from_spins(s, G) == 2
    assert sqrt_int(49) == 7.0
    assert (to_pm1(np.array([0, 1])) == np.array([-1, 1])).all()


# ------------------------------------------------------------------ #
#  CIM wrapper: extract_spins & silence_stdio                        #
# ------------------------------------------------------------------ #
def test_extract_spins_from_mock():
    spins = np.array([[1, -1, 1], [-1, -1, 1]])
    mock = types.SimpleNamespace(spin_configurations=spins)
    out = extract_spins(mock, 3)
    assert out.shape == spins.shape
    assert set(out.flat) == {-1, 1}


def test_silence_stdio():
    buf = io.StringIO()
    with silence_stdio():
        print("this will be muted")
    print("kept", file=buf, end="")
    assert buf.getvalue() == "kept"

