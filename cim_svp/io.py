from pathlib import Path
import ast, re, warnings
import numpy as np
import yaml

__all__ = ["load_lattice", "save_yaml"]

_DIGIT_RE = re.compile(r"-?\d+")


def _tokenise(line: str):
    """Return all integers in *line* (list may be empty)."""
    return [int(x) for x in _DIGIT_RE.findall(line)]


def _validate_square(mat: np.ndarray, dim: int):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Basis must be square.")
    if mat.shape[0] != dim:
        raise ValueError(f"Expected dimension {dim}, got {mat.shape[0]}.")


def load_lattice(
    dim: int,
    seed: int,
    folder: str = "svp_lattices",
    *,
    fmt: str = "auto"  # "auto" (default) | "challenge" | "literal"
) -> np.ndarray:
    """
    Load a dim×dim lattice basis.

    fmt = "auto"       : try challenge format first; if it fails,
                         fall back to Python literal **and emit a warning**.
    fmt = "challenge"  : force challenge tokeniser only.
    fmt = "literal"    : force Python literal only.
    """
    path = Path(folder) / f"dim{dim}_seed{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text()

    def parse_challenge():
        rows = [_tokenise(ln) for ln in text.splitlines()]
        rows = [r for r in rows if r]              # drop lines w/o digits
        if len(rows) != dim or {len(r) for r in rows} != {dim}:
            raise ValueError("challenge parser shape mismatch")
        return np.array(rows, dtype=object)

    def parse_literal():
        try:
            mat = np.array(ast.literal_eval(text), dtype=object)
        except Exception as exc:
            raise ValueError("literal parser failed") from exc
        _validate_square(mat, dim)
        return mat

    if fmt == "challenge":
        return parse_challenge()
    if fmt == "literal":
        return parse_literal()
    if fmt == "auto":
        try:
            return parse_challenge()
        except ValueError:
            warnings.warn(
                "load_lattice: falling back to Python-literal parser "
                "because challenge parser failed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return parse_literal()

    raise ValueError("fmt must be 'auto', 'challenge', or 'literal'")


def save_yaml(obj, path):
    Path(path).write_text(yaml.dump(obj, sort_keys=False))
