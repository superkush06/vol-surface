"""SVI — Stochastic Volatility Inspired raw parameterisation (Gatheral 2004).

Total implied variance w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where k = log(K/F).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
