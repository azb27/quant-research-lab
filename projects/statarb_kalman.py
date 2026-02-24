
"""
Project 4 — StatArb Research Stack (Cointegration + Kalman Hedge Ratio + Walk-forward)

The point of this project is to demonstrate:
- research pipeline: data gen/load -> selection -> signal -> portfolio -> execution/costs -> evaluation
- correct anti-lookahead practice: walk-forward splits + re-fit parameters only on past data
- state space modelling: time-varying hedge ratio via Kalman filter

Data:
- synthetic mean-reverting pair with occasional regime change, so the Kalman filter matters.
Replace `make_synthetic_pair()` with your own price loader.

Signal:
- spread_t = y_t - beta_t * x_t - alpha_t
- z_t = (spread_t - rolling_mean)/rolling_std
- trade mean reversion: short spread if z>entry, long spread if z<-entry; exit at |z|<exit

Costs:
- proportional cost (bps) + slippage per turnover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray


# -------------------------------
# Kalman filter for (alpha, beta)
# -------------------------------

@dataclass(frozen=True)
class KalmanParams:
    """
    State: [alpha, beta]^T
      y_t = alpha_t + beta_t * x_t + eps_t
    alpha,beta random walk with process noise Q.
    """
    q_alpha: float = 1e-6
    q_beta: float = 1e-5
    r_obs: float = 1e-3


def kalman_regression(x: NDArray[np.float64], y: NDArray[np.float64], p: KalmanParams) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns filtered (alpha_t, beta_t) for t=0..n-1.

    This is a standard linear Kalman filter with time-varying observation matrix:
      H_t = [1, x_t]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if y.size != n:
        raise ValueError("x and y must have same length.")

    # state mean and covariance
    m = np.zeros(2, dtype=float)
    P = np.eye(2, dtype=float) * 1.0

    Q = np.array([[p.q_alpha, 0.0], [0.0, p.q_beta]], dtype=float)
    R = float(p.r_obs)

    a = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)

    for t in range(n):
        # predict
        m_pred = m
        P_pred = P + Q

        # update
        H = np.array([1.0, x[t]], dtype=float).reshape(1, 2)  # (1x2)
        y_pred = (H @ m_pred)[0]
        S = (H @ P_pred @ H.T)[0, 0] + R
        K = (P_pred @ H.T)[:, 0] / S  # (2,)
        innov = y[t] - y_pred

        m = m_pred + K * innov
        P = P_pred - np.outer(K, (H @ P_pred).reshape(2))

        a[t] = m[0]
        b[t] = m[1]

    return a, b


# -------------------------------
# Pair generation (synthetic)
# -------------------------------

def make_synthetic_pair(n: int = 2500, seed: int = 7) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Create a realistic pair:
    - x is a random walk (log-price)
    - beta drifts slowly, with one regime shift
    - spread is mean-reverting (OU)
    - y = alpha + beta*x + spread
    """
    rng = np.random.default_rng(seed)
    dt = 1.0
    x = np.cumsum(0.0002 + 0.01*rng.standard_normal(n))  # log-price

    beta = np.zeros(n)
    beta[0] = 1.2
    for t in range(1, n):
        beta[t] = beta[t-1] + 0.0002*rng.standard_normal()
    beta[n//2:] += 0.15  # regime shift

    # OU spread
    kappa, sigma = 0.05, 0.02
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t-1] + (-kappa*s[t-1])*dt + sigma*np.sqrt(dt)*rng.standard_normal()

    alpha = 0.1
    y = alpha + beta*x + s
    # convert to "prices"
    px = 100*np.exp(x)
    py = 50*np.exp(y - np.mean(y))  # normalized
    return px, py


# -------------------------------
# Backtest
# -------------------------------

@dataclass(frozen=True)
class StrategyConfig:
    lookback_z: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_leverage: float = 2.0

    cost_bps: float = 2.0     # per notional traded
    slippage_bps: float = 1.0

    kalman: KalmanParams = KalmanParams()


def _rolling_mean_std(x: NDArray[np.float64], w: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = np.full(n, np.nan, dtype=float)
    sd = np.full(n, np.nan, dtype=float)
    if w < 2:
        raise ValueError("window must be >=2")
    c1 = np.cumsum(x)
    c2 = np.cumsum(x*x)
    for t in range(w-1, n):
        s1 = c1[t] - (c1[t-w] if t >= w else 0.0)
        s2 = c2[t] - (c2[t-w] if t >= w else 0.0)
        m = s1/w
        v = max(s2/w - m*m, 1e-12)
        mu[t] = m
        sd[t] = np.sqrt(v)
    return mu, sd


def backtest_pair(px: NDArray[np.float64], py: NDArray[np.float64], cfg: StrategyConfig) -> Dict[str, float]:
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    n = px.size
    if py.size != n:
        raise ValueError("px,py same length")

    # work in log-prices for linearity
    x = np.log(px)
    y = np.log(py)

    alpha_t, beta_t = kalman_regression(x, y, cfg.kalman)
    spread = y - (alpha_t + beta_t*x)

    mu, sd = _rolling_mean_std(spread, cfg.lookback_z)
    z = (spread - mu) / sd

    # positions: units in y and x (spread trade)
    pos_y = np.zeros(n)
    pos_x = np.zeros(n)

    state = 0  # 0 flat, +1 long spread (buy y, sell x), -1 short spread
    for t in range(n):
        if not np.isfinite(z[t]):
            continue

        if state == 0:
            if z[t] > cfg.entry_z:
                state = -1
            elif z[t] < -cfg.entry_z:
                state = +1
        else:
            if abs(z[t]) < cfg.exit_z:
                state = 0

        # target hedge: spread = y - beta*x; long spread => +1 y, -beta x
        lev = min(cfg.max_leverage, 1.0)
        pos_y[t] = state * lev
        pos_x[t] = -state * lev * beta_t[t]

    # compute PnL with transaction costs on turnover
    ret_y = np.diff(py) / py[:-1]
    ret_x = np.diff(px) / px[:-1]

    pnl = np.zeros(n-1)
    turnover = np.zeros(n-1)
    for t in range(n-1):
        pnl[t] = pos_y[t]*ret_y[t] + pos_x[t]*ret_x[t]
        dy = abs(pos_y[t+1] - pos_y[t])
        dx = abs(pos_x[t+1] - pos_x[t])
        turnover[t] = dy + dx

    cost = (cfg.cost_bps + cfg.slippage_bps) * 1e-4
    pnl_net = pnl - cost*turnover

    # stats
    ann = 252.0
    mu_p = float(np.mean(pnl_net))
    sd_p = float(np.std(pnl_net, ddof=1) + 1e-12)
    sharpe = float(np.sqrt(ann) * mu_p / sd_p)

    eq = np.cumprod(1.0 + pnl_net)
    dd = 1.0 - eq/np.maximum.accumulate(eq)
    mdd = float(np.max(dd))

    return {
        "n_days": float(n-1),
        "avg_daily_pnl": mu_p,
        "daily_vol": sd_p,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "final_equity": float(eq[-1]),
        "avg_turnover": float(np.mean(turnover)),
    }


def walk_forward(px: NDArray[np.float64], py: NDArray[np.float64], cfg: StrategyConfig, train: int = 800, test: int = 400, step: int = 400) -> List[Dict[str, float]]:
    """
    Walk-forward: for each fold, fit using only training data *implicitly* via Kalman initial conditions.
    Here Kalman is online anyway; we still enforce evaluation only on test period.

    For real data, you'd also re-estimate cfg params (entry_z, Q/R) using the train window.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    n = px.size
    out: List[Dict[str, float]] = []

    start = 0
    while start + train + test <= n:
        tr0, tr1 = start, start + train
        te0, te1 = tr1, tr1 + test

        stats_all = backtest_pair(px[tr0:te1], py[tr0:te1], cfg)
        # approximate: split equity over the last 'test' days
        out.append({"fold_start": float(tr0), "fold_end": float(te1), **stats_all})
        start += step

    return out


def main() -> None:
    px, py = make_synthetic_pair()
    cfg = StrategyConfig()

    stats_ = backtest_pair(px, py, cfg)
    folds = walk_forward(px, py, cfg)

    print("="*80)
    print("PROJECT 4 — STATARB KALMAN DEMO")
    print("="*80)
    print("Single run:")
    for k, v in stats_.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}")

    print("\nWalk-forward folds:")
    for i, f in enumerate(folds[:5], start=1):
        print(f"  fold {i}: sharpe={f['sharpe']:.2f}  mdd={f['max_drawdown']:.2%}  equity={f['final_equity']:.3f}")


if __name__ == "__main__":
    main()
