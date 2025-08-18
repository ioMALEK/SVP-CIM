"""
cim_svp.hotpatch
================
Work-around for the UnboundLocalError in cim-optimizer 1.0.4
when use_CAC=False (AHD mode).  Safe to import multiple times.
"""

import cim_optimizer.solve_Ising as _si

def _patch_isolated_print_bug():
    if getattr(_si, "_ahd_patch_applied", False):
        return                                      # already done
    original = _si.Ising.solver.solve

    def wrapped(self, *a, **k):
        try:
            return original(self, *a, **k)
        except UnboundLocalError as err:
            if "best_ahc_ext_lambd" in str(err):
                # Skip the faulty print and return the object (self).
                return self
            raise                                     # propagate unrelated errors

    _si.Ising.solver.solve = wrapped
    _si._ahd_patch_applied = True                    # idempotent flag

_patch_isolated_print_bug()