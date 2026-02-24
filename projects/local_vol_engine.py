
"""
Project 3 — Local Volatility (SVI → Dupire → Local-Vol MC)

Pipeline:
1) Fit SVI slices on each maturity to market implied vols
2) Build a smooth total-variance surface w(K,T) (linear interpolation in w)
3) Compute Dupire local vol σ_loc(K,T) on a grid using finite differences
4) Interpolate σ_loc(S,t) for Monte Carlo path simulation
5) Price exotics (digitals, barriers) under local vol

Notes:
- In production you'd add stronger static arbitrage constraints; here we keep
  a pragmatic penalty + diagnostic checks.
- Local vol is a deterministic function; it fits vanillas but does NOT model
  smile dynamics. It's a strong baseline for exotics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm


# -------------------------------
# SVI slice
# -------------------------------

@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y, dtype=float)
        return self.a + self.b * (self.rho*(y - self.m) + np.sqrt((y - self.m)**2 + self.sigma**2))

    def implied_vol(self, y: NDArray[np.float64], T: float) -> NDArray[np.float64]:
        w = self.total_variance(y)
        return np.sqrt(np.maximum(w / max(T, 1e-12), 1e-12))


def calibrate_svi(y: NDArray[np.float64], iv: NDArray[np.float64], T: float) -> SVIParams:
    y = np.asarray(y, dtype=float)
    iv = np.asarray(iv, dtype=float)
    w_mkt = iv*iv*T
    w_atm = float(np.interp(0.0, y, w_mkt))

    weights = np.ones_like(y) / max(len(y), 1)

    def objective(p: NDArray[np.float64]) -> float:
        a, b, rho, m, sig = map(float, p)
        if b <= 0 or sig <= 0 or not (-0.999 < rho < 0.999):
            return 1e6
        svi = SVIParams(a, b, rho, m, sig)
        w = svi.total_variance(y)
        fit = float(np.sum(weights*(w - w_mkt)**2))
        pen = 50.0*float(np.sum(np.maximum(-w, 0.0)**2))  # w>=0
        return fit + pen

    bounds = [(-0.5, 0.8), (1e-4, 4.0), (-0.99, 0.99), (-1.0, 1.0), (1e-4, 2.0)]
    x0 = np.array([w_atm, 0.2, -0.3, 0.0, 0.2], dtype=float)

    res_g = optimize.differential_evolution(objective, bounds=bounds, seed=1, maxiter=120, tol=1e-10, popsize=12)
    res_l = optimize.minimize(objective, res_g.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    a, b, rho, m, sig = map(float, res_l.x)
    return SVIParams(a, b, rho, m, sig)


# -------------------------------
# Implied vol surface (w-interp)
# -------------------------------

class ImpliedVolSurface:
    def __init__(self, S0: float, r: float):
        self.S0 = float(S0)
        self.r = float(r)
        self._slices: Dict[float, SVIParams] = {}
        self.expiries: List[float] = []

    def fit(self, market: List[Dict[str, float]]) -> None:
        from collections import defaultdict
        byT = defaultdict(list)
        for d in market:
            byT[float(d["T"])].append(d)

        self.expiries = sorted(byT.keys())
        for T in self.expiries:
            F = self.S0*np.exp(self.r*T)
            y = np.array([np.log(float(x["K"])/F) for x in byT[T]], dtype=float)
            iv = np.array([float(x["iv"]) for x in byT[T]], dtype=float)
            self._slices[T] = calibrate_svi(y, iv, T)

    def total_variance(self, K: float, T: float) -> float:
        T = float(T)
        if T <= self.expiries[0]:
            T0 = self.expiries[0]
            F0 = self.S0*np.exp(self.r*T0)
            y0 = np.log(float(K)/F0)
            return float(self._slices[T0].total_variance(np.array([y0]))[0])
        if T >= self.expiries[-1]:
            T1 = self.expiries[-1]
            F1 = self.S0*np.exp(self.r*T1)
            y1 = np.log(float(K)/F1)
            return float(self._slices[T1].total_variance(np.array([y1]))[0])

        idx = np.searchsorted(self.expiries, T) - 1
        T1, T2 = self.expiries[idx], self.expiries[idx+1]

        F1 = self.S0*np.exp(self.r*T1)
        F2 = self.S0*np.exp(self.r*T2)
        y1 = np.log(float(K)/F1)
        y2 = np.log(float(K)/F2)

        w1 = float(self._slices[T1].total_variance(np.array([y1]))[0])
        w2 = float(self._slices[T2].total_variance(np.array([y2]))[0])

        a = (T - T1)/(T2 - T1)
        return (1-a)*w1 + a*w2

    def implied_vol(self, K: float, T: float) -> float:
        w = self.total_variance(K, T)
        return float(np.sqrt(max(w/max(T, 1e-12), 1e-12)))


# -------------------------------
# Dupire local vol on a grid
# -------------------------------

class DupireLocalVolGrid:
    """
    Build σ_loc(K,T) on a (T,K) grid using Dupire on call prices derived from the IV surface.
    Then interpolate σ_loc(S,t) by evaluating at K=S (standard Markov mapping).
    """
    def __init__(self, iv_surface: ImpliedVolSurface, S0: float, r: float):
        self.iv = iv_surface
        self.S0 = float(S0)
        self.r = float(r)

        self.K_grid: Optional[NDArray[np.float64]] = None
        self.T_grid: Optional[NDArray[np.float64]] = None
        self.sigma_grid: Optional[NDArray[np.float64]] = None
        self._interp: Optional[RegularGridInterpolator] = None

    def _bs_call(self, K: float, T: float) -> float:
        sig = self.iv.implied_vol(K, T)
        if T <= 0:
            return max(self.S0 - K, 0.0)
        d1 = (np.log(self.S0/K) + (self.r + 0.5*sig*sig)*T)/(sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return float(self.S0*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2))

    def _dupire_at(self, K: float, T: float) -> float:
        if T < 1e-3:
            return self.iv.implied_vol(K, max(T, 1e-3))

        dK = 0.005*K
        dT = max(0.01*T, 0.005)

        C_Tp = self._bs_call(K, T + dT)
        C_Tm = self._bs_call(K, max(T - dT, 1e-3))
        dC_dT = (C_Tp - C_Tm)/(2*dT)

        C_Kp = self._bs_call(K + dK, T)
        C_Km = self._bs_call(max(K - dK, 1e-6), T)
        C_K = self._bs_call(K, T)
        dC_dK = (C_Kp - C_Km)/(2*dK)
        d2C_dK2 = (C_Kp - 2*C_K + C_Km)/(dK*dK)

        num = dC_dT + self.r*K*dC_dK
        den = 0.5*K*K*d2C_dK2
        if num <= 0 or den <= 0:
            return self.iv.implied_vol(K, T)
        return float(np.sqrt(max(num/den, 1e-10)))

    def build(self, K_grid: NDArray[np.float64], T_grid: NDArray[np.float64]) -> None:
        K_grid = np.asarray(K_grid, dtype=float)
        T_grid = np.asarray(T_grid, dtype=float)
        sig = np.zeros((T_grid.size, K_grid.size), dtype=float)

        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                sig[i, j] = self._dupire_at(float(K), float(T))

        self.K_grid = K_grid
        self.T_grid = T_grid
        self.sigma_grid = sig
        self._interp = RegularGridInterpolator((T_grid, K_grid), sig, bounds_error=False, fill_value=None)

    def sigma_loc(self, S: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """
        Evaluate σ_loc(S,t) by interpolating σ_loc(K=tbd,T=t) at K=S.
        """
        if self._interp is None:
            raise RuntimeError("Call build() first.")
        S = np.asarray(S, dtype=float)
        tt = float(max(t, float(self.T_grid[0] if self.T_grid is not None else 1e-3)))
        pts = np.column_stack([np.full(S.size, tt), S.reshape(-1)])
        out = self._interp(pts)
        return np.asarray(out, dtype=float)


# -------------------------------
# Local vol Monte Carlo
# -------------------------------

class LocalVolMC:
    def __init__(self, lv: DupireLocalVolGrid, S0: float, r: float, seed: int = 42):
        self.lv = lv
        self.S0 = float(S0)
        self.r = float(r)
        self.rng = np.random.default_rng(seed)

    def simulate(self, T: float, n_steps: int, n_paths: int) -> NDArray[np.float64]:
        dt = float(T)/n_steps
        sqrt_dt = np.sqrt(dt)
        S = np.full(n_paths, self.S0, dtype=float)
        paths = np.zeros((n_paths, n_steps+1), dtype=float)
        paths[:, 0] = S

        for i in range(n_steps):
            t_mid = (i + 0.5)*dt
            sig = np.maximum(self.lv.sigma_loc(S, t_mid), 1e-6)
            Z = self.rng.standard_normal(n_paths)
            S = S*np.exp((self.r - 0.5*sig*sig)*dt + sig*sqrt_dt*Z)
            paths[:, i+1] = S
        return paths

    def price_european(self, K: float, T: float, is_call: bool, n_steps: int = 200, n_paths: int = 100_000) -> Dict[str, float]:
        paths = self.simulate(T, n_steps, n_paths)
        ST = paths[:, -1]
        pay = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        disc = np.exp(-self.r*T)
        x = disc*pay
        return {"price": float(np.mean(x)), "std_error": float(np.std(x, ddof=1)/np.sqrt(n_paths))}

    def price_digital(self, K: float, T: float, is_call: bool, n_steps: int = 200, n_paths: int = 100_000) -> Dict[str, float]:
        paths = self.simulate(T, n_steps, n_paths)
        ST = paths[:, -1]
        pay = (ST > K).astype(float) if is_call else (ST < K).astype(float)
        disc = np.exp(-self.r*T)
        x = disc*pay
        return {"price": float(np.mean(x)), "std_error": float(np.std(x, ddof=1)/np.sqrt(n_paths))}


# -------------------------------
# Synthetic market surface
# -------------------------------

def generate_market_surface(S0: float, r: float) -> List[Dict[str, float]]:
    rng = np.random.default_rng(0)
    expiries = [0.083, 0.167, 0.25, 0.5, 1.0, 2.0]
    moneyness = np.linspace(0.75, 1.25, 21)
    out: List[Dict[str, float]] = []

    for T in expiries:
        F = S0*np.exp(r*T)
        base = 0.20
        for m in moneyness:
            K = S0*m
            y = np.log(K/F)
            skew = -0.12*y/np.sqrt(T + 0.1)
            smile = 0.08*(y*y)/(T + 0.2)
            term = -0.01*np.sqrt(T)
            iv = max(0.05, base + skew + smile + term + rng.normal(0, 0.001))
            out.append({"K": float(K), "T": float(T), "iv": float(iv)})
    return out


# -------------------------------
# Demo
# -------------------------------

def main() -> None:
    S0, r = 100.0, 0.03
    market = generate_market_surface(S0, r)

    ivs = ImpliedVolSurface(S0, r)
    ivs.fit(market)

    K_grid = np.linspace(70, 130, 121)
    T_grid = np.linspace(0.02, 2.0, 61)

    lv = DupireLocalVolGrid(ivs, S0, r)
    lv.build(K_grid, T_grid)

    mc = LocalVolMC(lv, S0, r, seed=42)

    print("="*80)
    print("PROJECT 3 — LOCAL VOL ENGINE DEMO")
    print("="*80)
    print(f"SVI expiries fitted: {ivs.expiries}")

    T = 0.5
    for K in [85, 90, 100, 110, 115]:
        pr = mc.price_european(K, T, True, n_steps=250, n_paths=120_000)
        dg = mc.price_digital(K, T, True, n_steps=250, n_paths=120_000)
        print(f"K={K:>5.1f}  Call={pr['price']:.4f} (SE={pr['std_error']:.4f})  Digital={dg['price']:.4f}")

    # quick sanity: local vol around ATM near 0.5y should be in the same ballpark as IV
    iv_atm = ivs.implied_vol(S0, T)
    lv_atm = float(lv.sigma_loc(np.array([S0]), T)[0])
    print(f"\nATM IV(T=0.5): {iv_atm:.2%}   ATM local vol(T=0.5): {lv_atm:.2%}")


if __name__ == "__main__":
    main()
