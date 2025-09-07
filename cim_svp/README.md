# `cim_svp` helper package

Reusable helpers used for SVP studies concerning CIM/CPM. 

```
cim_svp/
├─ __init__.py      (re-exports the public API)
├─ io.py            (lattice I/O + YAML helper)
├─ maths.py         (exact integer maths, Ising mapping)
├─ cimwrap.py       (CIM wrappers: spin extraction, silencer)
└─ cpmwrap.py       (Same but for CPM)
```
---

## Summary of main ones

### `io.py`
| function | description |
|----------|-------------|
| `load_lattice(dim, seed, folder="svp_lattices")` | Parse a basis file (Python literal *or* plain integers) into a NumPy array with `dtype=object`. |
| `save_yaml(obj, path)` | One-liner wrapper around `yaml.dump`, used for manifests and best-param dumps. |

### `maths.py`
| function | description |
|----------|-------------|
| `vec_sq_norm_int(v)` | Exact ‖v‖² for arbitrary-precision integers. |
| `gram_int(B)` | Exact Gram matrix `B Bᵀ`. |
| `build_J(B, scale=1.0)` | Safe float64 Ising matrix `J = –G/max|G|·scale`, diag = 0. |
| `sq_from_spins(s, G_int)` | Exact quadratic form `sᵀ G s`, `s∈{±1}^K`. |
| `sqrt_int(n)` | Accurate √n as float, even for 200-digit ints. |
| `to_pm1(arr)` | Convert `{0,1}` or `{−1,0,1}` arrays to strict ±1. |


### `cimwrap.py`
| symbol | description |
|--------|-------------|
| `silence_stdio()` | Context manager that mutes CIM console messages. |
| `extract_spins(result, K)` | Robustly extracts an `(R×K)` ±1 matrix from any `cim_optimizer` result object/dict. |
| `safe_solve()` | Placeholder wrapper for future time-out / retry logic (currently just calls `solver.solve` inside `silence_stdio`). |

---

## Unit tests

Run `pytest -q`; tests live in `tests/` and cover

---

*End of file*
