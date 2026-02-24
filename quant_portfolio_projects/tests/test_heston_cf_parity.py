
import numpy as np
from projects.heston_engine import HestonParams, HestonCF

def test_put_call_parity():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    r = 0.03
    pr = HestonCF(p, r)
    S, K, T = 100.0, 100.0, 0.5
    C = pr.call(S, K, T)
    P = pr.put(S, K, T)
    parity = C - P - (S - K*np.exp(-r*T))
    assert abs(parity) < 5e-4
