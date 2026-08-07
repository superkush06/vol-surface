"""Implied volatility via Brent's method (pure Python, no scipy)."""

from __future__ import annotations

import math
from collections.abc import Callable

from .black_scholes import BlackScholes


class IVSolverError(RuntimeError):
    """Implied vol could not be solved (e.g., price below intrinsic)."""


def implied_vol(market_price: float, bs: BlackScholes, *,
                call: bool = True, tol: float = 1e-8, max_iter: int = 100,
                lo: float = 1e-6, hi: float = 5.0) -> float:
    """Find sigma such that bs.price(sigma, call) == market_price.

    Uses Brent's method on [lo, hi]. Raises IVSolverError if the market
    price is outside the no-arbitrage range.
    """
    intrinsic = (
        max(bs.S * math.exp(-bs.q * bs.T) - bs.K * math.exp(-bs.r * bs.T), 0.0)
        if call else
        max(bs.K * math.exp(-bs.r * bs.T) - bs.S * math.exp(-bs.q * bs.T), 0.0)
    )
    upper_arb = (
        bs.S * math.exp(-bs.q * bs.T) if call else bs.K * math.exp(-bs.r * bs.T)
    )
    if market_price < intrinsic - tol or market_price > upper_arb + tol:
        raise IVSolverError(
            f"market price {market_price} is outside arbitrage bounds "
            f"[{intrinsic}, {upper_arb}]"
        )

    def f(s: float) -> float:
        return bs.price(s, call=call) - market_price
    a, b = lo, hi
    fa, fb = f(a), f(b)
    # A bracket end can *be* the root — deep out of the money the price is flat
    # in sigma to the last bit, so f(lo) is exactly zero. Brent needs a sign
    # change and would reject that as a bad bracket.
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        if fb < 0:
            raise IVSolverError(
                f"implied vol exceeds the upper bracket hi={hi}: the option "
                f"price at sigma={hi} is still below the market price "
                f"(short-dated / event vols can do this — pass a larger `hi`)"
            )
        raise IVSolverError(
            f"implied vol is below the lower bracket lo={lo}: the option "
            f"price at sigma={lo} already exceeds the market price "
            f"(pass a smaller `lo`)"
        )
    return _brentq(f, a, b, fa=fa, fb=fb, tol=tol, max_iter=max_iter)


def _brentq(f: Callable[[float], float], a: float, b: float, *, fa: float | None = None,
            fb: float | None = None, tol: float = 1e-8,
            max_iter: int = 100) -> float:
    """Brent's method root finder (1973). Assumes f(a)*f(b) < 0.

    `fa`/`fb` may be supplied to reuse already-computed endpoint values.
    """
    fa = f(a) if fa is None else fa
    fb = f(b) if fb is None else fb
    if fa * fb >= 0:
        raise IVSolverError(f"f(a)*f(b) >= 0: {fa=}, {fb=}")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    d = b - a
    mflag = True
    for _ in range(max_iter):
        if fb == 0 or abs(b - a) < tol:
            return b
        if fa != fc and fb != fc:
            # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # secant
            s = b - fb * (b - a) / (fb - fa)
        # a and b are not kept ordered (only |f(b)| <= |f(a)|), so the
        # acceptance interval between (3a+b)/4 and b needs explicit bounds.
        lo_i, hi_i = sorted(((3 * a + b) / 4.0, b))
        cond = (
            (s < lo_i or s > hi_i) or
            (mflag and abs(s - b) >= abs(b - c) / 2) or
            (not mflag and abs(s - b) >= abs(c - d) / 2) or
            (mflag and abs(b - c) < tol) or
            (not mflag and abs(c - d) < tol)
        )
        if cond:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    raise IVSolverError(f"brentq did not converge in {max_iter} iterations")
