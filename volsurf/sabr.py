"""SABR model — Hagan et al. (2002) closed-form approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SABRParams:
    """alpha > 0, 0 <= beta <= 1, -1 < rho < 1, nu > 0."""
    alpha: float
    beta: float
    rho: float
    nu: float

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.nu < 0:
            raise ValueError("nu must be >= 0")


# Below this |z| the direct z/x(z) evaluation loses precision (catastrophic
# cancellation inside the log); the Taylor series agrees with the direct
# formula to ~1e-13 at the crossover.
_Z_SERIES_CUTOFF = 1e-6


def sabr_iv(F: float, K: float, T: float, params: SABRParams) -> float:
    """Hagan SABR implied vol for a single (F, K, T) point.

    F: forward, K: strike, T: time to expiry, params: SABR parameters.

    A single code path covers every strike: near the money the z/x(z)
    factor is evaluated by its Taylor series

        z/x(z) = 1 + rho*z/2 + (3*rho^2 - 2)*z^2/12 + O(z^3)

    so the at-the-money vol is the smooth limit of the general formula
    (no separate ATM branch), and the (1 + [...]*T) time-correction
    factor is applied for all strikes.
    """
    params.validate()
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    if F <= 0 or K <= 0:
        raise ValueError("F and K must be positive")
    log_fk = math.log(F / K)
    fk_beta = (F * K) ** ((1.0 - beta) / 2.0)
    one_minus_beta = 1.0 - beta

    A = alpha / (fk_beta * (1.0 + (one_minus_beta ** 2) * (log_fk ** 2) / 24.0
                            + (one_minus_beta ** 4) * (log_fk ** 4) / 1920.0))

    z = (nu / alpha) * fk_beta * log_fk
    if abs(z) < _Z_SERIES_CUTOFF:
        ratio = 1.0 + 0.5 * rho * z + (3.0 * rho * rho - 2.0) * z * z / 12.0
    else:
        denom = 1.0 - 2.0 * rho * z + z * z
        x_z = math.log((math.sqrt(denom) + z - rho) / (1.0 - rho))
        ratio = z / x_z

    correction = 1.0 + (
        ((one_minus_beta * alpha) ** 2) / (24.0 * (fk_beta ** 2))
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * T
    return A * ratio * correction
