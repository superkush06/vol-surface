"""No-arbitrage checks on the IV smile.

Butterfly arbitrage: d^2 C / dK^2 >= 0 — densities must be non-negative.
Calendar arbitrage: w(k, T) must be non-decreasing in T at each k.
"""

from __future__ import annotations


def total_variance(iv: float, T: float) -> float:
    return iv * iv * T


def butterfly_violations(strikes: list[float], ivs: list[float], T: float,
                         eps: float = 1e-6) -> list[int]:
    """Indices of strikes where the implied density is negative.

    Approximates d^2C/dK^2 by central difference of the BS-implied price.
    """
    from .black_scholes import BlackScholes
    if len(strikes) < 3:
        return []
    violations = []
    for i in range(1, len(strikes) - 1):
        K_minus, K, K_plus = strikes[i - 1], strikes[i], strikes[i + 1]
        # Use unit forward = K_minus for relative scaling; absolute spot does
        # not affect the sign of d^2 C / dK^2 in this scan.
        S = K
        f_minus = BlackScholes(S, K_minus, T).price(ivs[i - 1])
        f_zero  = BlackScholes(S, K,       T).price(ivs[i])
        f_plus  = BlackScholes(S, K_plus,  T).price(ivs[i + 1])
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
    out = []
    for i, (sv, lv) in enumerate(zip(ivs_short, ivs_long, strict=False)):
        if total_variance(lv, T_long) + eps < total_variance(sv, T_short):
            out.append(i)
    return out
