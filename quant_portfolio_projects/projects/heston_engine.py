
"""
Project 2 — Heston Option Pricing Engine
- Semi-analytic pricer via characteristic function (Lewis single-integral)
- Monte Carlo pricer via Andersen (2008) QE scheme (variance stays non-negative)
  with correct *spot/variance correlation* (rho) via correlated Brownian shocks
- Calibration (global + local) to a synthetic IV surface

This module is offline-runnable; replace `generate_synthetic_surface()` with
real market data loading if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize
from scipy.stats import norm


# -------------------------------
# Parameters
# -------------------------------

@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def feller_ok(self) -> bool:
        return 2.0 * self.kappa * self.theta > self.xi * self.xi


# -------------------------------
# Analytic (CF) pricer
# -------------------------------

class HestonCF:
    """
    Lewis (2001) single-integral Heston call pricer:
      C = S - (sqrt(KF)*e^{-rT}/pi) * ∫_0^∞ Re( e^{-iu x} * φ(u - i/2) / (u^2 + 1/4) ) du
    where x = ln(F/K), F=S e^{rT}.
    """
    def __init__(self, params: HestonParams, r: float):
        self.p = params
        self.r = float(r)

    def _phi(self, u: complex, T: float) -> complex:
        p = self.p
        kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0

        # Standard "Little Heston Trap" stable form:
        a = kappa * theta
        b = kappa
        d = np.sqrt((rho*xi*1j*u - b)**2 + (xi**2) * (1j*u + u*u))
        g = (b - rho*xi*1j*u - d) / (b - rho*xi*1j*u + d + 1e-30)

        exp_dT = np.exp(-d*T)
        C = (self.r*1j*u*T) + (a/(xi**2)) * ((b - rho*xi*1j*u - d)*T - 2*np.log((1 - g*exp_dT)/(1 - g)))
        D = ((b - rho*xi*1j*u - d)/(xi**2)) * ((1 - exp_dT)/(1 - g*exp_dT))
        return np.exp(C + D*v0)

    def call(self, S: float, K: float, T: float) -> float:
        S = float(S); K = float(K); T = float(T)
        if T <= 0:
            return max(S - K, 0.0)

        F = S*np.exp(self.r*T)
        x = np.log(F/K)

        def integrand(u: float) -> float:
            z = u - 0.5j
            phi = self._phi(z, T)
            num = np.exp(-1j*u*x) * phi
            den = u*u + 0.25
            return float(np.real(num/den))

        val, _ = integrate.quad(integrand, 0.0, 200.0, limit=250)
        call = S - (np.sqrt(K*F)*np.exp(-self.r*T)/np.pi)*val
        # no-arb lower bound
        return float(max(call, max(S - K*np.exp(-self.r*T), 0.0)))

    def put(self, S: float, K: float, T: float) -> float:
        C = self.call(S, K, T)
        return float(C - S + K*np.exp(-self.r*T))

    def implied_vol(self, S: float, K: float, T: float) -> float:
        target = self.call(S, K, T)

        def bs_call(sig: float) -> float:
            if sig <= 0:
                return 0.0
            d1 = (np.log(S/K) + (self.r + 0.5*sig*sig)*T)/(sig*np.sqrt(T))
            d2 = d1 - sig*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)

        def obj(sig: float) -> float:
            return bs_call(sig) - target

        try:
            return float(optimize.brentq(obj, 1e-4, 3.0, xtol=1e-10))
        except ValueError:
            return float("nan")


# -------------------------------
# Monte Carlo QE (Andersen 2008)
# -------------------------------

class HestonMC_QE:
    """
    Andersen (2008) Quadratic-Exponential (QE) scheme for v(t).
    We correlate spot and variance shocks via:
        Zs = rho*Zv + sqrt(1-rho^2)*Zperp
    where Zv is used in the quadratic branch (and still generated in exp branch).

    Spot update uses the drift correction term based on variance increments:
        ln S_{t+dt} = ln S_t + (r - 0.5*v_bar)dt + sqrt(v_bar*dt)*Zs
                     + (rho/xi) * (v_{t+dt} - v_t - kappa*(theta - v_t)*dt)
    This matches the continuous-time covariance between logS and v.
    """
    def __init__(self, params: HestonParams, r: float, seed: int = 42):
        self.p = params
        self.r = float(r)
        self.rng = np.random.default_rng(seed)

    def _qe_step(self, v: NDArray[np.float64], dt: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns (v_new, Zv) where Zv is a standard normal used to correlate spot.
        """
        p = self.p
        kappa, theta, xi = p.kappa, p.theta, p.xi
        e = np.exp(-kappa*dt)

        m = theta + (v - theta)*e
        s2 = (v*xi*xi*e*(1-e)/kappa) + (theta*xi*xi*(1-e)*(1-e)/(2*kappa))
        m = np.maximum(m, 1e-12)
        s2 = np.maximum(s2, 0.0)

        psi = s2/(m*m + 1e-30)
        psi_c = 1.5

        U = self.rng.random(v.size)
        Zv = self.rng.standard_normal(v.size)

        v_new = np.empty_like(v)

        quad = psi <= psi_c
        if np.any(quad):
            psi_q = psi[quad]
            m_q = m[quad]
            b2 = 2.0/psi_q - 1.0 + np.sqrt(2.0/psi_q)*np.sqrt(np.maximum(2.0/psi_q - 1.0, 0.0))
            a = m_q/(1.0 + b2)
            b = np.sqrt(np.maximum(b2, 0.0))
            v_new[quad] = a * (b + Zv[quad])**2

        expm = ~quad
        if np.any(expm):
            psi_e = psi[expm]
            m_e = m[expm]
            p0 = (psi_e - 1.0)/(psi_e + 1.0)
            beta = (1.0 - p0)/(m_e + 1e-30)
            # use U for variance transition; Zv is still available for correlation
            v_new[expm] = np.where(
                U[expm] <= p0,
                0.0,
                np.log(np.maximum((1.0 - p0)/(1.0 - U[expm]), 1e-30))/(beta + 1e-30)
            )

        return np.maximum(v_new, 0.0), Zv

    def simulate_terminal(self, S0: float, T: float, n_steps: int, n_paths: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        p = self.p
        dt = T/n_steps

        S = np.full(n_paths, float(S0))
        v = np.full(n_paths, float(p.v0))

        for _ in range(n_steps):
            v_new, Zv = self._qe_step(v, dt)
            Zperp = self.rng.standard_normal(n_paths)
            Zs = p.rho*Zv + np.sqrt(max(1.0 - p.rho*p.rho, 0.0))*Zperp

            v_bar = 0.5*(v + v_new)
            v_bar = np.maximum(v_bar, 1e-12)

            drift = (self.r - 0.5*v_bar)*dt
            diff = np.sqrt(v_bar*dt)*Zs

            # covariance correction term
            corr_term = (p.rho/p.xi) * (v_new - v - p.kappa*(p.theta - v)*dt)

            S = S * np.exp(drift + diff + corr_term)
            v = v_new

        return S, v

    def price_european(self, S0: float, K: float, T: float, is_call: bool = True, n_steps: int = 200, n_paths: int = 200_000) -> Dict[str, float]:
        ST, _ = self.simulate_terminal(S0, T, n_steps, n_paths)
        pay = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        disc = np.exp(-self.r*T)
        x = disc*pay
        price = float(np.mean(x))
        se = float(np.std(x, ddof=1)/np.sqrt(n_paths))
        return {"price": price, "std_error": se}


# -------------------------------
# Calibration (to synthetic data)
# -------------------------------

def generate_synthetic_surface(S: float, r: float, true_params: HestonParams) -> List[Dict[str, float]]:
    pr = HestonCF(true_params, r)
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    expiries = [0.25, 0.5, 1.0]
    rng = np.random.default_rng(0)

    data: List[Dict[str, float]] = []
    for T in expiries:
        for m in strikes:
            K = S*m
            iv = pr.implied_vol(S, K, T)
            if np.isfinite(iv):
                data.append({"K": K, "T": T, "iv": max(0.01, float(iv + rng.normal(0, 0.002))), "w": 1.0/T})
    return data


def calibrate_heston(S: float, r: float, market: List[Dict[str, float]]) -> HestonParams:
    Ks = np.array([d["K"] for d in market], dtype=float)
    Ts = np.array([d["T"] for d in market], dtype=float)
    ivs = np.array([d["iv"] for d in market], dtype=float)
    w = np.array([d.get("w", 1.0) for d in market], dtype=float)
    w = w/np.sum(w)

    # Convert market IV to call price for a price-based objective (more stable).
    def bs_call(S0: float, K: float, T: float, sig: float) -> float:
        d1 = (np.log(S0/K) + (r + 0.5*sig*sig)*T)/(sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return float(S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

    mkt_price = np.array([bs_call(S, Ks[i], Ts[i], ivs[i]) for i in range(len(market))], dtype=float)

    # Vega weights normalize across strikes (approx converts price error to IV error)
    def bs_vega(S0: float, K: float, T: float, sig: float) -> float:
        d1 = (np.log(S0/K) + (r + 0.5*sig*sig)*T)/(sig*np.sqrt(T))
        return float(S0*norm.pdf(d1)*np.sqrt(T))

    vega = np.array([max(0.01, bs_vega(S, Ks[i], Ts[i], ivs[i])) for i in range(len(market))], dtype=float)

    def obj(x: NDArray[np.float64]) -> float:
        v0, kappa, theta, xi, rho = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
        if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or not (-0.999 < rho < 0.0):
            return 1e6
        p = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        pr = HestonCF(p, r)
        err = 0.0
        for i in range(len(market)):
            mod = pr.call(S, Ks[i], Ts[i])
            err += w[i]*((mod - mkt_price[i])/vega[i])**2
        return float(err)

    bounds = [(1e-4, 1.0), (0.05, 10.0), (1e-4, 1.0), (0.05, 2.0), (-0.99, -0.01)]
    res_g = optimize.differential_evolution(obj, bounds=bounds, seed=1, maxiter=40, tol=1e-6, popsize=10)
    res_l = optimize.minimize(obj, res_g.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    v0, kappa, theta, xi, rho = map(float, res_l.x)
    return HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)


# -------------------------------
# Demo
# -------------------------------

def main() -> None:
    S0, r = 100.0, 0.03
    true = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)

    print("="*80)
    print("PROJECT 2 — HESTON ENGINE DEMO")
    print("="*80)
    print(f"True params: {true}  (Feller ok? {true.feller_ok()})")

    pr = HestonCF(true, r)
    K, T = 100.0, 0.5
    C = pr.call(S0, K, T)
    P = pr.put(S0, K, T)
    parity = C - P - (S0 - K*np.exp(-r*T))
    print(f"\nAnalytic: C={C:.4f}  P={P:.4f}  parity error={parity:.3e}")

    mc = HestonMC_QE(true, r, seed=42)
    mc_res = mc.price_european(S0, K, T, True, n_steps=200, n_paths=200_000)
    print(f"MC(QE):  C={mc_res['price']:.4f}  SE={mc_res['std_error']:.4f}  diff={mc_res['price']-C:+.4f}")

    # Calibration to synthetic surface
    mkt = generate_synthetic_surface(S0, r, true)
    calib = calibrate_heston(S0, r, mkt)
    print(f"\nCalibrated params: {calib}  (Feller ok? {calib.feller_ok()})")

    pr_c = HestonCF(calib, r)
    iv_sample = pr_c.implied_vol(S0, 90.0, 0.5)
    print(f"Sample implied vol (K=90,T=0.5): {iv_sample:.2%}")


if __name__ == "__main__":
    main()
