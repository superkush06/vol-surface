"""Multi-expiry SVI surface: per-slice calibration + calendar no-arbitrage.

The single-slice SVI in `svi.py` fits one expiry. A *surface* stitches several
expiries together and must additionally be free of **calendar arbitrage**: at
every log-moneyness k, total implied variance w(k, T) must be non-decreasing in
T (otherwise a calendar spread is a risk-free profit, Gatheral 2004).

This module adds:
  - `fit_svi_slice`  — least-squares raw-SVI fit to one (k, w) slice (no scipy),
  - `SVISurface`     — a set of fitted slices with T-interpolated variance,
  - `fit_svi_surface`— fit every slice and assemble the surface,
plus a calendar-arbitrage check across the fitted expiries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .svi import SVIRawParams, svi_w


# --- small self-contained Nelder-Mead (R^5, unconstrained) -----------------
def _nelder_mead(f, x0, *, max_iter: int = 800, tol: float = 1e-10):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] = v[i] * 1.05 + 0.05
        simplex.append(v)
    fs = [f(v) for v in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fs[i])
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]
        if fs[-1] - fs[0] < tol:
            break
        cen = [sum(simplex[j][d] for j in range(n)) / n for d in range(n)]
        worst = simplex[-1]
        refl = [cen[d] + (cen[d] - worst[d]) for d in range(n)]
        fr = f(refl)
        if fs[0] <= fr < fs[-2]:
            simplex[-1], fs[-1] = refl, fr
            continue
        if fr < fs[0]:
            exp = [cen[d] + 2.0 * (cen[d] - worst[d]) for d in range(n)]
            fe = f(exp)
            simplex[-1], fs[-1] = (exp, fe) if fe < fr else (refl, fr)
            continue
        con = [cen[d] + 0.5 * (worst[d] - cen[d]) for d in range(n)]
        fc = f(con)
        if fc < fs[-1]:
            simplex[-1], fs[-1] = con, fc
            continue
        best = simplex[0]
        simplex = [[best[d] + 0.5 * (v[d] - best[d]) for d in range(n)]
                   for v in simplex]
        fs = [f(v) for v in simplex]
    return simplex[0], fs[0]


def _x_to_params(x) -> SVIRawParams:
    a, log_b, atanh_rho, m, log_sigma = x
    return SVIRawParams(a=a, b=math.exp(log_b), rho=math.tanh(atanh_rho),
                        m=m, sigma=math.exp(log_sigma))


def _params_to_x(p: SVIRawParams):
    rho = max(-0.999, min(0.999, p.rho))
    return [p.a, math.log(max(p.b, 1e-8)), math.atanh(rho), p.m,
            math.log(max(p.sigma, 1e-8))]


def fit_svi_slice(ks: list[float], total_vars: list[float], *,
                  max_iter: int = 800) -> SVIRawParams:
    """Least-squares raw-SVI fit of one expiry slice.

    `ks` are log-moneyness, `total_vars` the observed total implied variance
    w = σ²T at those strikes. Reparameterised so b>0, |rho|<1, sigma>0 hold by
    construction; optimised with Nelder-Mead.
    """
    if len(ks) != len(total_vars) or len(ks) < 5:
        raise ValueError("need >=5 aligned (k, w) points")
    init = SVIRawParams(a=min(total_vars), b=0.1, rho=-0.3, m=0.0, sigma=0.1)

    def obj(x):
        p = _x_to_params(x)
        return sum((svi_w(k, p) - w) ** 2 for k, w in zip(ks, total_vars, strict=False))

    x_best, _ = _nelder_mead(obj, _params_to_x(init), max_iter=max_iter)
    return _x_to_params(x_best)


@dataclass
class SVISurface:
    """A calibrated multi-expiry SVI surface (expiries kept sorted)."""
    expiries: list[float]
    params: list[SVIRawParams]

    def __post_init__(self) -> None:
        order = sorted(range(len(self.expiries)), key=lambda i: self.expiries[i])
        self.expiries = [self.expiries[i] for i in order]
        self.params = [self.params[i] for i in order]

    def total_variance(self, k: float, T: float) -> float:
        """w(k, T) with linear-in-T interpolation between fitted slices."""
        ts = self.expiries
        if T <= ts[0]:
            return svi_w(k, self.params[0])
        if T >= ts[-1]:
            return svi_w(k, self.params[-1])
        for i in range(len(ts) - 1):
            if ts[i] <= T <= ts[i + 1]:
                w0, w1 = svi_w(k, self.params[i]), svi_w(k, self.params[i + 1])
                frac = (T - ts[i]) / (ts[i + 1] - ts[i])
                return w0 + frac * (w1 - w0)
        return svi_w(k, self.params[-1])

    def iv(self, k: float, T: float) -> float:
        w = self.total_variance(k, T)
        if w <= 0 or T <= 0:
            raise ValueError("non-positive total variance or expiry")
        return math.sqrt(w / T)

    def calendar_arbitrage_free(self, ks: list[float] | None = None,
                                tol: float = 1e-9) -> bool:
        """True iff total variance is non-decreasing in T at every test k."""
        if ks is None:
            ks = [-1.0 + 0.1 * i for i in range(21)]
        for k in ks:
            ws = [svi_w(k, p) for p in self.params]
            if any(ws[i + 1] < ws[i] - tol for i in range(len(ws) - 1)):
                return False
        return True


def fit_svi_surface(slices: list[tuple[float, list[float], list[float]]]
                    ) -> SVISurface:
    """Fit a surface from `(T, ks, total_vars)` slices, one per expiry."""
    slices = sorted(slices, key=lambda s: s[0])
    return SVISurface(
        expiries=[T for T, _, _ in slices],
        params=[fit_svi_slice(ks, w) for _, ks, w in slices],
    )
