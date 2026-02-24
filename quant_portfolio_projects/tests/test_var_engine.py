
import numpy as np
from projects.var_engine import var_es, kupiec_pof_test

def test_var_es_monotone():
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal(20000)
    v95, es95 = var_es(pnl, 0.95)
    v99, es99 = var_es(pnl, 0.99)
    assert v99 >= v95
    assert es99 >= es95
    assert es95 >= v95 - 1e-9

def test_kupiec_edge_case_zero_exceptions():
    exc = np.zeros(100, dtype=bool)
    out = kupiec_pof_test(exc, alpha=0.99)
    assert 0.0 <= out["p_value"] <= 1.0
