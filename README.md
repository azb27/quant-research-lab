# Quant Research Lab

Production-style quantitative research portfolio covering risk modeling, stochastic volatility pricing, volatility surface construction, and systematic trading research.

All modules are fully unit-tested and CI-validated.

---

## What This Demonstrates

- Monte Carlo simulation under correlated and heavy-tailed distributions  
- Stochastic volatility pricing (Heston CF + QE scheme)  
- Volatility surface construction (SVI → Dupire local volatility)  
- Dynamic state-space modeling (Kalman filter hedge ratios)  
- Backtesting with transaction costs and walk-forward validation  
- Numerical stability, vectorization, and reproducibility  

---

## Projects

### 1. Portfolio Risk Engine

VaR / ES engine with multiple scenario models:

- Historical Simulation  
- Correlated GBM Monte Carlo  
- Student-t Copula shocks  
- EVT / GPD tail smoothing  

Includes:

- Kupiec unconditional coverage test  
- Christoffersen independence test  

---

### 2. Heston Stochastic Volatility Engine

- Semi-analytic pricing via characteristic function (Lewis formulation)  
- Andersen (2008) Quadratic-Exponential Monte Carlo scheme  
- Proper spot–variance correlation handling  
- Calibration via global + local optimization  

---

### 3. Local Volatility Surface

- SVI slice calibration per maturity  
- Total variance interpolation (calendar-arbitrage aware)  
- Dupire local volatility extraction via finite differences  
- Local-vol Monte Carlo pricing for digitals  

---

### 4. Statistical Arbitrage Research Stack

- Kalman filter dynamic hedge ratio  
- Cointegration-based spread modeling  
- Z-score mean-reversion signals  
- Transaction cost modeling  
- Walk-forward validation  

---

## Quick Start

```bash
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m pytest -q

python -m projects.var_engine
python -m projects.heston_engine
python -m projects.local_vol_engine
python -m projects.statarb_kalman
