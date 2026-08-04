"""SVI — Stochastic Volatility Inspired raw parameterisation (Gatheral 2004).

Total implied variance w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where k = log(K/F).

Also implements the Gatheral-Jacquier (2014) butterfly-arbitrage machinery:
`svi_g` (the g(k) function whose non-negativity is equivalent to a
non-negative implied density) and `svi_density` (the risk-neutral density
of log-moneyness implied by the slice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ButterflyArbitrageWarning(UserWarning):
    """SVI parameters admit butterfly arbitrage: g(k) < 0 somewhere."""


@dataclass
class SVIRawParams:
    """Raw SVI parameters.

    a: vertical translation (>= 0 typically)
    b: slope (>= 0)
    rho: skew (-1 < rho < 1)
    m: horizontal translation
    sigma: smoothness (> 0)
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def validate(self) -> None:
        if self.b < 0:
            raise ValueError("b must be >= 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")


def svi_w(k: float, p: SVIRawParams) -> float:
    """Total implied variance w(k) for a single log-moneyness k."""
    p.validate()
    centered = k - p.m
    return p.a + p.b * (p.rho * centered +
                        math.sqrt(centered * centered + p.sigma * p.sigma))


def svi_iv(k: float, T: float, p: SVIRawParams) -> float:
    """Implied vol from total-variance via w = sigma_iv^2 * T."""
    w = svi_w(k, p)
    if w <= 0 or T <= 0:
        raise ValueError("non-positive total variance or expiry")
    return math.sqrt(w / T)


def _svi_w_derivs(k: float, p: SVIRawParams) -> tuple[float, float, float]:
    """(w, w', w'') for raw SVI — both derivatives are closed-form."""
    centered = k - p.m
    root = math.sqrt(centered * centered + p.sigma * p.sigma)
    w = p.a + p.b * (p.rho * centered + root)
    w1 = p.b * (p.rho + centered / root)
    w2 = p.b * p.sigma * p.sigma / (root * root * root)
    return w, w1, w2


def svi_g(k: float, p: SVIRawParams) -> float:
    """Gatheral-Jacquier (2014) g(k) for a raw SVI slice.

        g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2 (1/w + 1/4) + w''/2

    The slice is free of butterfly arbitrage iff g(k) >= 0 for all k
    (given w > 0): g is proportional to the implied risk-neutral density.
    """
    p.validate()
    w, w1, w2 = _svi_w_derivs(k, p)
    if w <= 0:
        raise ValueError("total variance must be positive to evaluate g(k)")
    half_slope = 0.5 * w1
    return ((1.0 - k * half_slope / w) ** 2
            - half_slope * half_slope * (1.0 / w + 0.25)
            + 0.5 * w2)


def svi_density(k: float, p: SVIRawParams) -> float:
    """Risk-neutral density of log-moneyness k implied by the SVI slice.

        p(k) = g(k) / sqrt(2 pi w(k)) * exp(-d_-(k)^2 / 2),
        d_-(k) = -k / sqrt(w) - sqrt(w) / 2.

    Depends only on the total-variance slice w(k), not on T separately.
    Negative values are exactly the butterfly-arbitrage regions (g < 0).
    """
    w = svi_w(k, p)
    if w <= 0:
        raise ValueError("total variance must be positive to evaluate density")
    g = svi_g(k, p)
    sqrt_w = math.sqrt(w)
    d_minus = -k / sqrt_w - 0.5 * sqrt_w
    return g / math.sqrt(2.0 * math.pi * w) * math.exp(-0.5 * d_minus * d_minus)


def svi_min_g(p: SVIRawParams, k_min: float, k_max: float,
              *, n: int = 201) -> tuple[float, float]:
    """Smallest g(k) on a uniform n-point grid over [k_min, k_max].

    Returns `(min_g, k_at_min)`. Knowing *where* the butterfly condition is
    tightest is what tells you which strikes to distrust, so this returns the
    location alongside the value rather than a bare bool.
    """
    if k_max <= k_min:
        raise ValueError("k_max must be > k_min")
    if n < 2:
        raise ValueError("n must be >= 2")
    step = (k_max - k_min) / (n - 1)
    best_k = k_min
    best_g = math.inf
    for i in range(n):
        k = k_min + i * step
        g = svi_g(k, p)
        if g < best_g:
            best_g, best_k = g, k
    return best_g, best_k


def svi_butterfly_arbitrage_free(p: SVIRawParams, k_min: float, k_max: float,
                                 *, n: int = 201, tol: float = 0.0) -> bool:
    """True iff g(k) >= -tol on a uniform n-point grid over [k_min, k_max].

    Grid-based, so it can miss a violation narrower than the grid spacing;
    tighten `n` on slices with small `sigma` (a sharp elbow makes g dip in a
    correspondingly narrow window).
    """
    return svi_min_g(p, k_min, k_max, n=n)[0] >= -tol
