"""Multi-expiry SVI surface: per-slice calibration + calendar no-arbitrage.

The single-slice SVI in `svi.py` fits one expiry. A *surface* stitches several
expiries together and must additionally be free of **calendar arbitrage**: at
every log-moneyness k, total implied variance w(k, T) must be non-decreasing in
T (otherwise a calendar spread is a risk-free profit, Gatheral 2004).

This module adds:
  - `fit_svi_slice`  — quasi-explicit raw-SVI fit to one (k, w) slice
                       (De Marco/Zeliade reduction + multistart, no scipy),
  - `SVISurface`     — a set of fitted slices with T-interpolated variance,
  - `fit_svi_surface`— fit every slice and assemble the surface,
plus a calendar-arbitrage check across the fitted expiries and a
butterfly-arbitrage (Gatheral-Jacquier g(k)) diagnostic on every fit.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from .svi import (
    ButterflyArbitrageWarning,
    SVIRawParams,
    svi_butterfly_arbitrage_free,
    svi_g,
    svi_w,
)


# --- small self-contained Nelder-Mead (unconstrained) -----------------------
def _nelder_mead(f, x0, *, max_iter: int = 800, tol: float = 1e-12):
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


# --- quasi-explicit (De Marco / Zeliade 2009) inner problem ------------------
def _solve3(A, rhs):
    """3x3 linear solve, Gaussian elimination with partial pivoting."""
    M = [list(row) + [r] for row, r in zip(A, rhs, strict=True)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, 3):
            fct = M[r][col] / M[col][col]
            for c in range(col, 4):
                M[r][c] -= fct * M[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        s = M[r][3] - sum(M[r][c] * x[c] for c in range(r + 1, 3))
        x[r] = s / M[r][r]
    return x


def _inner_fit(ks, ws, m: float, sigma: float):
    """For fixed (m, sigma) the SVI objective is linear least squares.

    With y = (k - m)/sigma,  w = a + d*y + c*sqrt(y^2 + 1)  where
    d = b*rho*sigma and c = b*sigma, so (a, d, c) solve a 3x3 normal
    system exactly; the SVI shape constraints become c >= 0, |d| <= c.
    Returns (sse, a, d, c).
    """
    ys = [(k - m) / sigma for k in ks]
    zs = [math.sqrt(y * y + 1.0) for y in ys]
    n = float(len(ks))
    sy = sum(ys); sz = sum(zs)
    syy = sum(y * y for y in ys); szz = sum(z * z for z in zs)
    syz = sum(y * z for y, z in zip(ys, zs, strict=True))
    sw = sum(ws)
    swy = sum(w * y for w, y in zip(ws, ys, strict=True))
    swz = sum(w * z for w, z in zip(ws, zs, strict=True))
    sol = _solve3([[n, sy, sz], [sy, syy, syz], [sz, syz, szz]],
                  [sw, swy, swz])
    if sol is None:
        return math.inf, 0.0, 0.0, 0.0
    a, d, c = sol
    # project onto the admissible cone: b >= 0 and |rho| < 1
    c = max(c, 1e-12)
    d = max(-0.999 * c, min(0.999 * c, d))
    a = (sw - d * sy - c * sz) / n  # re-fit intercept after clamping
    sse = sum((a + d * y + c * z - w) ** 2
              for y, z, w in zip(ys, zs, ws, strict=True))
    return sse, a, d, c


def _adc_to_params(a: float, d: float, c: float, m: float,
                   sigma: float) -> SVIRawParams:
    b = c / sigma
    rho = 0.0 if c < 1e-11 else max(-0.999, min(0.999, d / c))
    return SVIRawParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def fit_svi_slice(ks: list[float], total_vars: list[float], *,
                  max_iter: int = 800, check_butterfly: bool = True,
                  butterfly_penalty: float = 0.0) -> SVIRawParams:
    """Least-squares raw-SVI fit of one expiry slice.

    `ks` are log-moneyness, `total_vars` the observed total implied variance
    w = σ²T at those strikes.

    Uses the quasi-explicit De Marco/Zeliade reduction: for fixed (m, sigma)
    the remaining parameters solve a linear least-squares problem exactly, so
    only a 2-D outer problem in (m, sigma) is searched (coarse grid + local
    Nelder-Mead from the best starts), followed by a full 5-D polish. This is
    robust on steep skews where a single-start 5-D search lands in local
    minima.

    If `check_butterfly` is true (default), emits `ButterflyArbitrageWarning`
    when the fitted slice has g(k) < 0 somewhere on the fit range — i.e. the
    returned parameters admit butterfly arbitrage. A positive
    `butterfly_penalty` additionally adds `penalty * sum(min(g, 0)^2)` over
    the fit range to the polish objective, pushing the fit toward
    arbitrage-free parameters at some cost in least-squares error.
    """
    if len(ks) != len(total_vars) or len(ks) < 5:
        raise ValueError("need >=5 aligned (k, w) points")
    ws = list(total_vars)
    k_lo, k_hi = min(ks), max(ks)
    span = max(k_hi - k_lo, 1e-3)

    def outer(x):
        m, log_sigma = x
        sigma = math.exp(max(-12.0, min(3.0, log_sigma)))
        return _inner_fit(ks, ws, m, sigma)[0]

    # coarse grid over (m, sigma), keep the best few starts
    grid = []
    for fm in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        for sg in (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8):
            x = [k_lo + fm * span, math.log(sg)]
            grid.append((outer(x), x))
    grid.sort(key=lambda t: t[0])

    best_x, best_loss = None, math.inf
    for _, x0 in grid[:3]:
        x, loss = _nelder_mead(outer, x0, max_iter=max_iter, tol=1e-16)
        if loss < best_loss:
            best_x, best_loss = x, loss

    m = best_x[0]
    sigma = math.exp(max(-12.0, min(3.0, best_x[1])))
    _, a, d, c = _inner_fit(ks, ws, m, sigma)
    start = _adc_to_params(a, d, c, m, sigma)

    # full 5-D polish from the quasi-explicit solution
    g_grid = [k_lo + (k_hi - k_lo) * i / 100.0 for i in range(101)]

    def polish_obj(x):
        p = _x_to_params(x)
        sse = sum((svi_w(k, p) - w) ** 2 for k, w in zip(ks, ws, strict=True))
        if butterfly_penalty > 0.0:
            pen = 0.0
            for k in g_grid:
                try:
                    g = svi_g(k, p)
                except ValueError:
                    return math.inf
                if g < 0.0:
                    pen += g * g
            sse += butterfly_penalty * pen
        return sse

    x_best, polished_loss = _nelder_mead(polish_obj, _params_to_x(start),
                                         max_iter=max_iter, tol=1e-16)
    fit = _x_to_params(x_best)
    if butterfly_penalty <= 0.0 and polished_loss > best_loss:
        fit = start  # polish should never make things worse; keep the QE fit

    if check_butterfly:
        try:
            min_g = min(svi_g(k, fit) for k in g_grid)
        except ValueError:
            min_g = -math.inf
        if min_g < 0.0:
            warnings.warn(
                f"fitted SVI slice admits butterfly arbitrage: "
                f"min g(k) = {min_g:.4g} < 0 on [{k_lo:.3g}, {k_hi:.3g}] "
                f"(consider butterfly_penalty > 0)",
                ButterflyArbitrageWarning,
                stacklevel=2,
            )
    return fit


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
        """w(k, T) with linear-in-T interpolation between fitted slices.

        Below the first fitted expiry, the first slice is scaled
        proportionally in T (w -> w * T/T1), so implied vol stays flat at
        the first slice's level instead of diverging as T -> 0. Above the
        last fitted expiry, the last slice is held constant.
        """
        ts = self.expiries
        if T <= 0.0:
            return 0.0
        if T <= ts[0]:
            return svi_w(k, self.params[0]) * (T / ts[0])
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

    def butterfly_arbitrage_free(self, k_min: float = -1.0, k_max: float = 1.0,
                                 *, n: int = 201, tol: float = 0.0) -> bool:
        """True iff every fitted slice has g(k) >= -tol on [k_min, k_max].

        The companion to `calendar_arbitrage_free`: a surface is only usable
        if it is free of *both* arbitrages, and the butterfly condition is a
        per-slice statement while the calendar condition couples the slices.
        """
        return all(svi_butterfly_arbitrage_free(p, k_min, k_max, n=n, tol=tol)
                   for p in self.params)


def fit_svi_surface(slices: list[tuple[float, list[float], list[float]]],
                    **fit_kwargs) -> SVISurface:
    """Fit a surface from `(T, ks, total_vars)` slices, one per expiry.

    Keyword arguments are forwarded to `fit_svi_slice`.
    """
    slices = sorted(slices, key=lambda s: s[0])
    return SVISurface(
        expiries=[T for T, _, _ in slices],
        params=[fit_svi_slice(ks, w, **fit_kwargs) for _, ks, w in slices],
    )
