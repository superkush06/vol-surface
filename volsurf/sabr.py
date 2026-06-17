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


def sabr_iv(F: float, K: float, T: float, params: SABRParams) -> float:
    """Hagan SABR implied vol for a single (F, K, T) point.

    F: forward, K: strike, T: time to expiry, params: SABR parameters.
    Special-cases the at-the-money strike where z/x(z) approaches 1.
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

    if abs(F - K) < 1e-12:
        # ATM closed form
        sig = (alpha / (F ** one_minus_beta)) * (1.0 + (
            ((one_minus_beta * alpha) ** 2) / (24.0 * (F ** (2.0 * one_minus_beta)))
            + 0.25 * rho * beta * nu * alpha / (F ** one_minus_beta)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * T)
        return sig

    z = (nu / alpha) * fk_beta * log_fk
    denom = 1.0 - 2.0 * rho * z + z * z
    x_z = math.log((math.sqrt(denom) + z - rho) / (1.0 - rho))
    if abs(x_z) < 1e-12:
        return A
    correction = 1.0 + (
        ((one_minus_beta * alpha) ** 2) / (24.0 * (fk_beta ** 2))
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * T
    return A * (z / x_z) * correction
