import hashlib
GLOBAL_SEED = 2025

def make_seed(tag: str, *keys) -> int:
    """
    Deterministic 32-bit seed from arbitrary (hashable) keys.
    """
    h = hashlib.blake2b((tag + repr(keys)).encode(),
                        digest_size=4).digest()
    return int.from_bytes(h, "little") ^ GLOBAL_SEED
