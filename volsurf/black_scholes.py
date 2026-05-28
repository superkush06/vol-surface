"""Black-Scholes pricing and Greeks (closed form, pure Python)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _phi(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class BlackScholes:
    """Black-Scholes-Merton parameters and pricing methods.

    All quantities are continuously-compounded:
      S: spot
      K: strike
      T: time to expiry in years
      r: risk-free rate
      q: dividend yield
      sigma: volatility (annualised stdev of log-returns)
    """

    S: float
    K: float
    T: float
    r: float = 0.0
    q: float = 0.0

    def _d1(self, sigma: float) -> float:
        if sigma <= 0 or self.T <= 0:
            raise ValueError("sigma and T must be positive for d1")
        return (math.log(self.S / self.K) +
                (self.r - self.q + 0.5 * sigma * sigma) * self.T) / (sigma * math.sqrt(self.T))

    def _d2(self, sigma: float) -> float:
        return self._d1(sigma) - sigma * math.sqrt(self.T)

    def price(self, sigma: float, call: bool = True) -> float:
        if self.T <= 0:
            payoff = max(self.S - self.K, 0.0) if call else max(self.K - self.S, 0.0)
            return payoff
        d1, d2 = self._d1(sigma), self._d2(sigma)
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)
        if call:
            return self.S * disc_q * _phi(d1) - self.K * disc_r * _phi(d2)
        return self.K * disc_r * _phi(-d2) - self.S * disc_q * _phi(-d1)

    def vega(self, sigma: float) -> float:
        d1 = self._d1(sigma)
        return self.S * math.exp(-self.q * self.T) * _phi_pdf(d1) * math.sqrt(self.T)

    def delta(self, sigma: float, call: bool = True) -> float:
        d1 = self._d1(sigma)
        if call:
            return math.exp(-self.q * self.T) * _phi(d1)
        return math.exp(-self.q * self.T) * (_phi(d1) - 1.0)

    def gamma(self, sigma: float) -> float:
        d1 = self._d1(sigma)
        return math.exp(-self.q * self.T) * _phi_pdf(d1) / (self.S * sigma * math.sqrt(self.T))
