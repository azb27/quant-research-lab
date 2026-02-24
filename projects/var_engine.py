
"""
Project 1 — Portfolio Risk Engine (VaR / CVaR / Backtesting)

Key design goals (portfolio-ready):
- clean data models (dataclasses)
- vectorized simulation + pricing (no per-path Python loops)
- multiple scenario models:
    * Historical Simulation (HS) on log-returns
    * Monte Carlo GBM with correlated normals (+ variance reduction)
    * Student-t copula shocks (fat tails)
- risk metrics:
    * VaR and ES/CVaR at multiple confidence levels
    * optional EVT / GPD tail smoothing for ES
- model validation:
    * Kupiec POF (unconditional coverage)
    * Christoffersen (1998) independence test (clustering)

This module is self-contained and runnable offline with synthetic demo data.

References (conceptual):
- Jorion, "Value at Risk"
- McNeil, Frey, Embrechts, "Quantitative Risk Management"
- Kupiec (1995), Christoffersen (1998)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# -------------------------------
# Data models
# -------------------------------

@dataclass(frozen=True)
class Asset:
    name: str
    spot: float
    notional: float          # currency notional invested in the asset (positive long, negative short)
    annual_vol: float
    annual_drift: float = 0.0


@dataclass(frozen=True)
class EuroOption:
    """
    European option position priced with Black–Scholes for *revaluation*.

    In real portfolios you would plug in your own pricer or vol surface.
    For risk, repricing is the gold standard (vs pure Greeks P&L).
    """
    underlying_idx: int
    strike: float
    maturity_years: float
    is_call: bool
    quantity: float          # contracts * multiplier (signed)
    implied_vol: float


@dataclass(frozen=True)
class RiskConfig:
    horizon_days: int = 10
    n_sims: int = 200_000
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)

    seed: Optional[int] = 42

    # variance reduction for normal MC
    use_antithetic: bool = True
    use_stratified: bool = True

    # copula tail thickness
    copula_df: float = 5.0

    # EVT / GPD tail smoothing for ES
    use_evt_tail_for_es: bool = False
    evt_threshold_quantile: float = 0.90  # fit GPD on worst 10% of losses


# -------------------------------
# Black–Scholes (vectorized)
# -------------------------------

def _bs_d1_d2(S: NDArray[np.float64], K: float, T: float, r: float, sigma: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    S = np.asarray(S, dtype=float)
    if T <= 0 or sigma <= 0:
        d = np.zeros_like(S)
        return d, d
    sqrtT = math.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price_vec(S: NDArray[np.float64], K: float, T: float, r: float, sigma: float, is_call: bool) -> NDArray[np.float64]:
    """
    Vectorized BS price for an array of spots S.
    """
    S = np.asarray(S, dtype=float)
    if T <= 0:
        return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    discK = K * math.exp(-r * T)
    if is_call:
        return S * stats.norm.cdf(d1) - discK * stats.norm.cdf(d2)
    return discK * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


# -------------------------------
# Scenario generators
# -------------------------------

def _check_corr(corr: NDArray[np.float64]) -> None:
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation must be square.")
    if not np.allclose(corr, corr.T, atol=1e-12):
        raise ValueError("Correlation must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
        raise ValueError("Correlation diagonal must be 1.")
    eig = np.linalg.eigvalsh(corr)
    if np.min(eig) < -1e-10:
        raise ValueError(f"Correlation not PSD (min eigenvalue={np.min(eig):.3e}).")


def _stratified_normals(rng: np.random.Generator, n: int, d: int) -> NDArray[np.float64]:
    """
    Stratified sampling on U(0,1) per dimension, then inverse CDF.
    """
    Z = np.empty((n, d), dtype=float)
    strata = np.arange(n, dtype=float)
    for j in range(d):
        u = (strata + rng.random(n)) / n
        z = stats.norm.ppf(u)
        rng.shuffle(z)  # remove artificial dependence across dimensions
        Z[:, j] = z
    return Z


def generate_correlated_shocks(
    rng: np.random.Generator,
    n: int,
    corr: NDArray[np.float64],
    method: Literal["normal", "t_copula"] = "normal",
    df: float = 5.0,
    antithetic: bool = True,
    stratified: bool = True,
) -> NDArray[np.float64]:
    """
    Returns shocks with covariance = corr (approximately), shape (n, d).

    - normal: correlated Gaussian shocks
    - t_copula: Student-t copula shocks (fatter tails, same linear corr)
    """
    _check_corr(corr)
    d = corr.shape[0]
    L = np.linalg.cholesky(corr)

    half = n // 2 if antithetic else n

    if stratified:
        Z = _stratified_normals(rng, half, d)
    else:
        Z = rng.standard_normal((half, d))

    if antithetic:
        Z = np.vstack([Z, -Z])[:n]

    if method == "normal":
        return Z @ L.T

    if method == "t_copula":
        # t-copula construction:
        # 1) generate correlated normals
        # 2) map to uniforms via Phi
        # 3) map to t-quantiles via t^{-1}
        X = Z @ L.T
        U = stats.norm.cdf(X)
        T = stats.t.ppf(U, df=df)
        # rescale to have unit variance per marginal:
        # Var(t_df) = df/(df-2) for df>2
        scale = math.sqrt((df - 2.0) / df)
        return T * scale

    raise ValueError(f"Unknown method={method}")


# -------------------------------
# Risk metrics + EVT tail
# -------------------------------

def var_es(pnl: NDArray[np.float64], alpha: float, use_evt_es: bool = False, evt_threshold_q: float = 0.90) -> Tuple[float, float]:
    """
    VaR/ES computed on P&L array (positive = profit, negative = loss).
    Convention here: VaR and ES are returned as **positive numbers** (loss magnitude).

    If use_evt_es=True, we fit a GPD to tail *losses* beyond a threshold and
    compute a smoother ES. This stabilizes ES for small sample sizes.
    """
    pnl = np.asarray(pnl, dtype=float)
    loss = -pnl  # positive loss
    var = float(np.quantile(loss, alpha))

    if not use_evt_es:
        tail = loss[loss >= var]
        es = float(np.mean(tail)) if tail.size > 0 else var
        return var, es

    # EVT: pick threshold u at q-quantile of loss
    u = float(np.quantile(loss, evt_threshold_q))
    exceed = loss[loss > u] - u
    if exceed.size < 50:
        tail = loss[loss >= var]
        es = float(np.mean(tail)) if tail.size > 0 else var
        return var, es

    # Fit GPD (shape xi, scale beta) to exceedances
    c, loc, scale = stats.genpareto.fit(exceed, floc=0.0)
    xi = float(c)
    beta = float(scale)

    # ES formula for GPD (xi < 1 required for finite mean)
    # For p in (0,1), quantile:
    # VaR_p = u + beta/xi * (( (1-p_u)/(1-p) )^xi - 1) where p_u = P(L<=u)
    # For ES_p:
    # ES_p = (VaR_p + (beta - xi*u)) / (1 - xi)  ??? careful; better compute on exceedance.
    # We'll use standard peaks-over-threshold ES:
    # ES_p = (VaR_p + (beta + xi*(VaR_p - u))) / (1 - xi)   for xi < 1
    p_u = float(np.mean(loss <= u))
    p = alpha
    if p <= p_u + 1e-12:
        tail = loss[loss >= var]
        es = float(np.mean(tail)) if tail.size > 0 else var
        return var, es

    if xi >= 1.0:
        tail = loss[loss >= var]
        es = float(np.mean(tail)) if tail.size > 0 else var
        return var, es

    # EVT-adjusted quantile at level p
    q = (1 - p) / max(1 - p_u, 1e-12)
    var_evt = u + (beta / xi) * (q**(-xi) - 1.0) if abs(xi) > 1e-8 else u + beta * math.log(1.0 / q)

    es_evt = (var_evt + (beta + xi * (var_evt - u)) / (1 - xi))
    return float(max(var_evt, var)), float(es_evt)


# -------------------------------
# Backtesting
# -------------------------------

def kupiec_pof_test(exceptions: NDArray[np.bool_], alpha: float) -> Dict[str, float]:
    """
    Kupiec (1995) unconditional coverage test.
    H0: exception rate = (1-alpha)
    """
    exc = np.asarray(exceptions, dtype=bool)
    n = exc.size
    x = int(np.sum(exc))
    p0 = 1.0 - alpha
    if n == 0:
        raise ValueError("Need non-empty exception series.")
    phat = x / n

    # Handle edge cases
    if x == 0 or x == n:
        p_value = float(stats.binomtest(x, n, p0).pvalue)
        return {"LR": float("nan"), "p_value": p_value, "n": float(n), "x": float(x), "phat": phat, "p0": p0}

    ll0 = (n - x) * math.log(1 - p0) + x * math.log(p0)
    ll1 = (n - x) * math.log(1 - phat) + x * math.log(phat)
    LR = -2.0 * (ll0 - ll1)
    p_value = float(1.0 - stats.chi2.cdf(LR, df=1))
    return {"LR": float(LR), "p_value": p_value, "n": float(n), "x": float(x), "phat": phat, "p0": p0}


def christoffersen_independence_test(exceptions: NDArray[np.bool_]) -> Dict[str, float]:
    """
    Christoffersen (1998) independence test for exception clustering.

    Build a 2-state Markov chain of exceptions and compare:
    - H0: independent (p01=p11=p)
    - H1: dependent (p01 != p11)
    """
    exc = np.asarray(exceptions, dtype=bool)
    if exc.size < 2:
        raise ValueError("Need at least 2 observations.")

    x00 = np.sum((~exc[:-1]) & (~exc[1:]))
    x01 = np.sum((~exc[:-1]) & (exc[1:]))
    x10 = np.sum((exc[:-1]) & (~exc[1:]))
    x11 = np.sum((exc[:-1]) & (exc[1:]))

    def safe_p(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    p01 = safe_p(x01, x00 + x01)
    p11 = safe_p(x11, x10 + x11)
    p = safe_p(x01 + x11, x00 + x01 + x10 + x11)

    def ll_bin(x1: float, x0: float, p_: float) -> float:
        if p_ in (0.0, 1.0):
            if (p_ == 0.0 and x1 > 0) or (p_ == 1.0 and x0 > 0):
                return -1e100
            return 0.0
        return x1 * math.log(p_) + x0 * math.log(1 - p_)

    ll1 = ll_bin(x01, x00, p01) + ll_bin(x11, x10, p11)
    ll0 = ll_bin(x01 + x11, x00 + x10, p)
    LR = -2.0 * (ll0 - ll1)
    p_value = float(1.0 - stats.chi2.cdf(LR, df=1))
    return {"LR": float(LR), "p_value": p_value, "p01": float(p01), "p11": float(p11), "p": float(p)}


# -------------------------------
# Portfolio risk engine
# -------------------------------

class PortfolioRiskEngine:
    def __init__(
        self,
        assets: List[Asset],
        corr: NDArray[np.float64],
        options: Optional[List[EuroOption]] = None,
        r: float = 0.02,
        cfg: RiskConfig = RiskConfig(),
    ) -> None:
        self.assets = assets
        self.options = options or []
        self.corr = np.asarray(corr, dtype=float)
        _check_corr(self.corr)
        if self.corr.shape[0] != len(self.assets):
            raise ValueError("corr dimension must match assets.")

        self.r = float(r)
        self.cfg = cfg
        self.dt = cfg.horizon_days / 252.0

        self._rng = np.random.default_rng(cfg.seed)

    def _equity_pnl(self, future_spots: NDArray[np.float64]) -> NDArray[np.float64]:
        spots0 = np.array([a.spot for a in self.assets], dtype=float)
        shares = np.array([a.notional / a.spot for a in self.assets], dtype=float)
        dS = future_spots - spots0[None, :]
        return dS @ shares

    def _options_pnl(self, future_spots: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Vectorized repricing per option (loop over options, but vectorized over paths).
        """
        if not self.options:
            return np.zeros(future_spots.shape[0], dtype=float)

        pnl = np.zeros(future_spots.shape[0], dtype=float)
        for opt in self.options:
            S0 = self.assets[opt.underlying_idx].spot
            ST = future_spots[:, opt.underlying_idx]
            T0 = opt.maturity_years
            T1 = max(T0 - self.dt, 0.0)
            price0 = float(bs_price_vec(np.array([S0], dtype=float), opt.strike, T0, self.r, opt.implied_vol, opt.is_call)[0])
            price1 = bs_price_vec(ST, opt.strike, T1, self.r, opt.implied_vol, opt.is_call)
            pnl += opt.quantity * (price1 - price0)
        return pnl

    def pnl_from_spots(self, future_spots: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._equity_pnl(future_spots) + self._options_pnl(future_spots)

    def simulate_spots(
        self,
        shock_model: Literal["normal", "t_copula"] = "normal",
    ) -> NDArray[np.float64]:
        """
        Correlated GBM:
            ln S_T = ln S_0 + (mu - 0.5 sigma^2) dt + sigma sqrt(dt) * eps
        """
        vols = np.array([a.annual_vol for a in self.assets], dtype=float)
        mus = np.array([a.annual_drift for a in self.assets], dtype=float)
        spots0 = np.array([a.spot for a in self.assets], dtype=float)

        eps = generate_correlated_shocks(
            rng=self._rng,
            n=self.cfg.n_sims,
            corr=self.corr,
            method=shock_model,
            df=self.cfg.copula_df,
            antithetic=self.cfg.use_antithetic,
            stratified=self.cfg.use_stratified,
        )

        log_ret = (mus - 0.5 * vols * vols) * self.dt + vols * math.sqrt(self.dt) * eps
        return spots0[None, :] * np.exp(log_ret)

    def risk_report_mc(self, shock_model: Literal["normal", "t_copula"] = "normal") -> Dict:
        spotsT = self.simulate_spots(shock_model=shock_model)
        pnl = self.pnl_from_spots(spotsT)

        out = {
            "model": f"MC_GBM_{shock_model}",
            "n_sims": int(self.cfg.n_sims),
            "horizon_days": int(self.cfg.horizon_days),
            "pnl_mean": float(np.mean(pnl)),
            "pnl_std": float(np.std(pnl, ddof=1)),
            "var": {},
            "es": {},
        }

        for cl in self.cfg.confidence_levels:
            alpha = cl
            var_, es_ = var_es(
                pnl=pnl,
                alpha=alpha,
                use_evt_es=self.cfg.use_evt_tail_for_es,
                evt_threshold_q=self.cfg.evt_threshold_quantile,
            )
            out["var"][f"{alpha:.0%}"] = float(var_)
            out["es"][f"{alpha:.0%}"] = float(es_)

        return out

    def risk_report_historical(self, historical_log_returns: NDArray[np.float64]) -> Dict:
        """
        Historical simulation on log-returns.

        Input shape: (n_obs, n_assets). For horizon_days>1 we aggregate by scaling
        with sqrt(h) (a simplification). A production HS would resample *blocks*.
        """
        R = np.asarray(historical_log_returns, dtype=float)
        if R.ndim != 2 or R.shape[1] != len(self.assets):
            raise ValueError("historical_log_returns must be (n_obs, n_assets).")
        spots0 = np.array([a.spot for a in self.assets], dtype=float)
        # scale to horizon (approx)
        R_h = R * math.sqrt(self.cfg.horizon_days)
        spotsT = spots0[None, :] * np.exp(R_h)
        pnl = self.pnl_from_spots(spotsT)

        out = {
            "model": "HistoricalSimulation",
            "n_obs": int(R.shape[0]),
            "horizon_days": int(self.cfg.horizon_days),
            "pnl_mean": float(np.mean(pnl)),
            "pnl_std": float(np.std(pnl, ddof=1)),
            "var": {},
            "es": {},
        }
        for cl in self.cfg.confidence_levels:
            var_, es_ = var_es(
                pnl=pnl,
                alpha=cl,
                use_evt_es=self.cfg.use_evt_tail_for_es,
                evt_threshold_q=self.cfg.evt_threshold_quantile,
            )
            out["var"][f"{cl:.0%}"] = float(var_)
            out["es"][f"{cl:.0%}"] = float(es_)
        return out

    @staticmethod
    def backtest_var(
        pnl_realized: NDArray[np.float64],
        var_level: float,
        alpha: float,
    ) -> Dict:
        """
        exceptions occur when loss > VaR  <=> pnl < -VaR
        """
        pnl_realized = np.asarray(pnl_realized, dtype=float)
        exc = pnl_realized < -float(var_level)

        kupiec = kupiec_pof_test(exc, alpha=alpha)
        christ = christoffersen_independence_test(exc)

        return {
            "alpha": float(alpha),
            "var_level": float(var_level),
            "n": int(exc.size),
            "exceptions": int(np.sum(exc)),
            "kupiec": kupiec,
            "christoffersen": christ,
        }


# -------------------------------
# Demo
# -------------------------------

def _synthetic_returns(n: int, corr: NDArray[np.float64], vol_daily: NDArray[np.float64], df: float = 6.0, seed: int = 7) -> NDArray[np.float64]:
    """
    Create heavy-tailed synthetic log-returns with a t-copula.
    """
    rng = np.random.default_rng(seed)
    d = corr.shape[0]
    eps = generate_correlated_shocks(rng, n, corr, method="t_copula", df=df, antithetic=False, stratified=False)
    return eps * vol_daily[None, :]


def main() -> None:
    # Portfolio: 3 assets + 1 SPX-like put
    assets = [
        Asset("SPY", spot=450.0, notional=500_000, annual_vol=0.18, annual_drift=0.06),
        Asset("TLT", spot=100.0, notional=200_000, annual_vol=0.15, annual_drift=0.02),
        Asset("GLD", spot=185.0, notional=100_000, annual_vol=0.16, annual_drift=0.02),
    ]
    corr = np.array([
        [1.00, -0.35, 0.05],
        [-0.35, 1.00, 0.20],
        [0.05, 0.20, 1.00],
    ], dtype=float)

    options = [
        EuroOption(underlying_idx=0, strike=430.0, maturity_years=0.25, is_call=False, quantity=1100.0, implied_vol=0.20),
    ]

    cfg = RiskConfig(
        horizon_days=10,
        n_sims=200_000,
        confidence_levels=(0.95, 0.99),
        seed=42,
        use_antithetic=True,
        use_stratified=True,
        copula_df=5.0,
        use_evt_tail_for_es=True,
        evt_threshold_quantile=0.90,
    )

    eng = PortfolioRiskEngine(assets, corr, options=options, r=0.03, cfg=cfg)
    rep_n = eng.risk_report_mc("normal")
    rep_t = eng.risk_report_mc("t_copula")

    # Synthetic "realized" PnL series to backtest against 99% VaR
    vol_daily = np.array([a.annual_vol for a in assets]) / math.sqrt(252.0)
    R_hist = _synthetic_returns(1500, corr, vol_daily, df=6.0, seed=123)
    hs_rep = eng.risk_report_historical(R_hist)

    var_99 = rep_t["var"]["99%"]
    pnl_real = hs_rep["pnl_mean"] + (hs_rep["pnl_std"] * np.random.default_rng(0).standard_normal(500))  # quick demo
    bt = PortfolioRiskEngine.backtest_var(pnl_real, var_99, alpha=0.99)

    print("=" * 80)
    print("PROJECT 1 — RISK ENGINE DEMO")
    print("=" * 80)
    for rep in (rep_n, rep_t, hs_rep):
        print(f"\nModel: {rep['model']}")
        for k, v in rep["var"].items():
            print(f"  {k} VaR: {v:,.2f}   ES: {rep['es'][k]:,.2f}")
    print("\nBacktest (99% VaR from t-copula MC on synthetic realized PnL):")
    print(f"  exceptions: {bt['exceptions']}/{bt['n']}")
    print(f"  Kupiec p-value: {bt['kupiec']['p_value']:.4f}")
    print(f"  Christoffersen p-value: {bt['christoffersen']['p_value']:.4f}")


if __name__ == "__main__":
    main()
