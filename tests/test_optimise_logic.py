from scripts.optimise_cim_static import optimise_dim


def test_patience_stops_fast():
    """Patience=3 ⇒ optimiser exits with stop_reason='patience'."""
    res = optimise_dim(dim=3, max_trials=50, patience=3)
    assert res["stop_reason"] == "patience"


def test_theoretical_bound_stops():
    """Ridiculously high bound should stop quickly."""
    res = optimise_dim(dim=3, max_trials=50, theoretical_bound=1e9)
    assert res["stop_reason"] == "theoretical_bound"