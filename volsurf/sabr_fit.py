"""SABR slice calibration: fit (alpha, rho, nu) to market IVs at fixed beta."""

from __future__ import annotations

import math

from .sabr import SABRParams, sabr_iv


def _objective(params: SABRParams, F: float, T: float,
               strikes: list[float], market_ivs: list[float],
               weights: list[float]) -> float:
    """Weighted sum of squared IV errors."""
    sse = 0.0
    for K, mv, w in zip(strikes, market_ivs, weights, strict=False):
        model = sabr_iv(F, K, T, params)
        sse += w * (model - mv) ** 2
    return sse


def fit_sabr(F: float, T: float, strikes: list[float], market_ivs: list[float],
             beta: float = 0.5, weights: list[float] | None = None,
             max_iter: int = 200, tol: float = 1e-8) -> SABRParams:
    """Grid-search initialisation + Nelder-Mead in 3D for (alpha, rho, nu).

    Pure Python, no scipy dep. Good enough for slice-level fits.
    """
    if weights is None:
        weights = [1.0] * len(strikes)
    if len(strikes) != len(market_ivs) or len(strikes) != len(weights):
        raise ValueError("strikes/market_ivs/weights length mismatch")

    # Initial guess by coarse grid
    best, best_loss = None, math.inf
    for a in (0.1, 0.2, 0.4, 0.8):
        for r in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for n in (0.1, 0.4, 0.8, 1.5):
                p = SABRParams(alpha=a, beta=beta, rho=r, nu=n)
                try:
                    loss = _objective(p, F, T, strikes, market_ivs, weights)
                except (ValueError, ZeroDivisionError):
                    continue
                if loss < best_loss:
                    best, best_loss = p, loss

    if best is None:
        raise RuntimeError("SABR initial grid found no valid point")
    return _nelder_mead(best, beta, F, T, strikes, market_ivs, weights, max_iter, tol)


def _params_to_x(p: SABRParams) -> list[float]:
    return [p.alpha, p.rho, p.nu]


def _x_to_params(x: list[float], beta: float) -> SABRParams:
    alpha = max(x[0], 1e-6)
    rho = max(-0.999, min(0.999, x[1]))
    nu = max(x[2], 1e-6)
    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)


def _nelder_mead(p0: SABRParams, beta: float, F: float, T: float,
                 K: list[float], IV: list[float], W: list[float],
                 max_iter: int, tol: float) -> SABRParams:
    """Pure-Python Nelder-Mead in R^3 — bounded via _x_to_params clamping."""
    x0 = _params_to_x(p0)
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] = v[i] * 1.05 + 0.05
        simplex.append(v)

    def loss(v):
        try:
            return _objective(_x_to_params(v, beta), F, T, K, IV, W)
        except (ValueError, ZeroDivisionError):
            return 1e9

    f = [loss(v) for v in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: f[i])
        simplex = [simplex[i] for i in order]
        f = [f[i] for i in order]
        if f[-1] - f[0] < tol:
            break
        centroid = [sum(v[d] for v in simplex[:-1]) / n for d in range(n)]
        worst = simplex[-1]
        reflected = [centroid[d] + (centroid[d] - worst[d]) for d in range(n)]
        f_r = loss(reflected)
        if f[0] <= f_r < f[-2]:
            simplex[-1] = reflected; f[-1] = f_r; continue
        if f_r < f[0]:
            expanded = [centroid[d] + 2.0 * (centroid[d] - worst[d]) for d in range(n)]
            f_e = loss(expanded)
            if f_e < f_r:
                simplex[-1] = expanded; f[-1] = f_e
            else:
                simplex[-1] = reflected; f[-1] = f_r
            continue
        contracted = [centroid[d] + 0.5 * (worst[d] - centroid[d]) for d in range(n)]
        f_c = loss(contracted)
        if f_c < f[-1]:
            simplex[-1] = contracted; f[-1] = f_c; continue
        best = simplex[0]
        simplex = [[best[d] + 0.5 * (v[d] - best[d]) for d in range(n)] for v in simplex]
        f = [loss(v) for v in simplex]
    return _x_to_params(simplex[0], beta)
