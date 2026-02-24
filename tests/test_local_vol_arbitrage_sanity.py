
import numpy as np
from projects.local_vol_engine import generate_market_surface, ImpliedVolSurface, DupireLocalVolGrid

def test_calendar_variance_non_decreasing():
    S0, r = 100.0, 0.03
    market = generate_market_surface(S0, r)
    ivs = ImpliedVolSurface(S0, r)
    ivs.fit(market)

    K = 100.0
    Ts = np.array([0.083, 0.25, 0.5, 1.0, 2.0])
    w = np.array([ivs.total_variance(K, float(T)) for T in Ts])
    assert np.all(np.diff(w) >= -1e-6)

def test_local_vol_positive():
    S0, r = 100.0, 0.03
    market = generate_market_surface(S0, r)
    ivs = ImpliedVolSurface(S0, r)
    ivs.fit(market)

    K_grid = np.linspace(70, 130, 51)
    T_grid = np.linspace(0.05, 1.0, 21)
    lv = DupireLocalVolGrid(ivs, S0, r)
    lv.build(K_grid, T_grid)
    sig = lv.sigma_loc(np.array([80.0, 100.0, 120.0]), 0.5)
    assert np.all(sig > 0)
