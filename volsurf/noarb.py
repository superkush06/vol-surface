"""Discrete no-arbitrage screens for raw quotes.

These are the model-free counterparts to the analytic checks in `svi.py`:
they take strikes and implied vols as quoted, with no fit in between.

Butterfly — by Breeden-Litzenberger, d^2C/dK^2 is the risk-neutral density,
so a negative second difference of the call curve is a butterfly spread with
non-negative payoff and negative cost. The second difference is taken across
prices at a *single* forward, which the caller must supply: the density
depends on where the distribution is centred, so pricing each strike triple
at its own middle strike answers a different question and can miss genuine
violations.

Calendar — total implied variance w(k, T) = sigma^2 T must be non-decreasing
in T at every log-moneyness, otherwise the calendar spread is free money.
Mismatched input lengths are an error rather than a silent truncation: an
arbitrage screen that skips strikes without saying so is worse than no
screen at all.
"""

from __future__ import annotations


def total_variance(iv: float, T: float) -> float:
    return iv * iv * T


def butterfly_violations(strikes: list[float], ivs: list[float], T: float,
                         forward: float, eps: float = 1e-6) -> list[int]:
    """Indices of strikes where the implied density is negative.

    Approximates d^2C/dK^2 by central difference of Black-Scholes prices,
    with every option priced at the same underlying `forward`. The forward
    matters: the implied density depends on where the distribution is
    centred, so pricing each strike triple at a different spot changes the
    sign of the finite difference and can miss genuine butterflies.
    """
    from .black_scholes import BlackScholes
    if len(strikes) != len(ivs):
        raise ValueError("strikes/ivs length mismatch")
    if forward <= 0:
        raise ValueError("forward must be positive")
    if len(strikes) < 3:
        return []
    violations = []
    for i in range(1, len(strikes) - 1):
        K_minus, K, K_plus = strikes[i - 1], strikes[i], strikes[i + 1]
        f_minus = BlackScholes(forward, K_minus, T).price(ivs[i - 1])
        f_zero  = BlackScholes(forward, K,       T).price(ivs[i])
        f_plus  = BlackScholes(forward, K_plus,  T).price(ivs[i + 1])
        h_minus = K - K_minus
        h_plus = K_plus - K
        # asymmetric 2nd derivative
        second = 2.0 * (
            f_minus / (h_minus * (h_minus + h_plus))
            - f_zero  / (h_minus * h_plus)
            + f_plus  / (h_plus  * (h_minus + h_plus))
        )
        if second < -eps:
            violations.append(i)
    return violations


def calendar_violations(strikes: list[float], ivs_short: list[float],
                        ivs_long: list[float], T_short: float, T_long: float,
                        eps: float = 1e-8) -> list[int]:
    """Indices where w(k, T_long) < w(k, T_short) — calendar arbitrage."""
    if T_long <= T_short:
        raise ValueError("T_long must be > T_short")
    if not len(strikes) == len(ivs_short) == len(ivs_long):
        raise ValueError(
            f"length mismatch: {len(strikes)} strikes, "
            f"{len(ivs_short)} short IVs, {len(ivs_long)} long IVs"
        )
    out = []
    for i, (sv, lv) in enumerate(zip(ivs_short, ivs_long, strict=True)):
        if total_variance(lv, T_long) + eps < total_variance(sv, T_short):
            out.append(i)
    return out
