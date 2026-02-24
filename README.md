# Quant Research / Trading Portfolio Projects (4)

These projects are designed to look like **production-quality quant research code**:
- modular structure
- deterministic runs via seeds
- numerical stability + vectorization
- clear docs + minimal dependencies
- unit tests for core math

## Projects

1) **Risk Engine — VaR / CVaR / Backtesting (HS, MC-GBM, Student-t Copula, EVT tail)**
   - Multi-asset portfolio (cash equities + European options)
   - Variance reduction (antithetic + stratified)
   - Backtests: Kupiec POF + Christoffersen independence test
   - Optional EVT/GPD tail for ES stability

2) **Option Pricing Engine — Heston (Analytic CF + QE Monte Carlo + calibration)**
   - Stable characteristic function integration
   - Andersen (2008) QE variance scheme with correct **rho correlation**
   - Control variate vs Black–Scholes for variance reduction
   - Calibration to a synthetic IV surface (easy to swap to real data)

3) **Local Vol — SVI surface → Dupire local vol → Local-Vol Monte Carlo**
   - Fit SVI slices per maturity
   - Build arbitrage-aware total-variance surface
   - Compute Dupire local vol on a grid and interpolate σ_loc(S,t)
   - Price digitals + barriers (sensitive exotics)

4) **StatArb Research Stack — Cointegration + Kalman hedge ratio + walk-forward**
   - Pair selection (correlation + Engle–Granger)
   - Dynamic hedge ratio via Kalman filter
   - Robust backtest with costs, slippage, position limits
   - Walk-forward training/validation to prevent lookahead

## How to run

```bash
pip install -r requirements.txt
python -m projects.var_engine
python -m projects.heston_engine
python -m projects.local_vol_engine
python -m projects.statarb_kalman
pytest -q
```

## Repo layout

```
quant_portfolio_projects/
  projects/
    var_engine.py
    heston_engine.py
    local_vol_engine.py
    statarb_kalman.py
  tests/
    test_var_engine.py
    test_heston_cf_parity.py
    test_local_vol_arbitrage_sanity.py
    test_kalman_filter.py
  requirements.txt
  README.md
```

## Notes
- Market data is **synthetic** so the code is runnable offline; swap in your own data loader later.
- Each module can be used as a library (import classes) or run as a script (demo in `main()`).
