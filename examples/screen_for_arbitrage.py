"""Three ways a volatility mark can be inadmissible, and how to catch each.

Run:  PYTHONPATH=. python3 examples/screen_for_arbitrage.py

1. Analytic butterfly:  g(k) on a fitted SVI slice — exact, closed form.
2. Discrete butterfly:  a finite-difference density on raw quotes, which
                        needs the forward to be stated, not guessed.
3. Calendar:            total variance that falls as maturity rises.

Every number printed here comes out of `volsurf`; nothing is hard-coded.
"""

from __future__ import annotations

import math
import warnings

from volsurf import (
    BlackScholes,
    ButterflyArbitrageWarning,
    SVIRawParams,
    butterfly_violations,
    calendar_violations,
    fit_svi_slice,
    svi_density,
    svi_min_g,
    svi_w,
)

# Axel Vogt's slice (Gatheral & Jacquier 2014): the canonical bad smile.
VOGT = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def analytic_butterfly() -> None:
    rule("1. g(k) on an SVI slice")
    min_g, k_star = svi_min_g(VOGT, -1.5, 1.5, n=3001)
    print(f"quoted slice           {VOGT}")
    print(f"min g(k)               {min_g:+.6f} at k = {k_star:.3f}")
    print(f"density there          {svi_density(k_star, VOGT):+.3e}"
          f"   <- a negative probability")

    ks = [-1.5 + 0.05 * i for i in range(61)]
    ws = [svi_w(k, VOGT) for k in ks]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ButterflyArbitrageWarning)
        fit_svi_slice(ks, ws)
    for w in caught:
        print(f"refitting warns        {w.message}")

    repaired = fit_svi_slice(ks, ws, butterfly_penalty=1e5)
    rep_g, _ = svi_min_g(repaired, -1.5, 1.5, n=3001)
    cost = max(abs(math.sqrt(svi_w(k, repaired)) - math.sqrt(svi_w(k, VOGT)))
               for k in ks)
    print(f"with a g(k) penalty    {repaired}")
    print(f"                       min g(k) = {rep_g:+.6f}, and the smile moves"
          f" at most {cost * 100:.2f} vol points")


def discrete_butterfly() -> None:
    rule("2. a finite-difference density on raw quotes")
    strikes = [70, 80, 90, 100, 110, 120, 130]
    ivs = [0.4308, 0.4065, 0.3218, 0.2787, 0.2548, 0.2533, 0.2780]
    print(f"strikes                {strikes}")
    print(f"implied vols           {ivs}")
    for forward in (85.0, 100.0, 120.0):
        bad = butterfly_violations(strikes, ivs, T=0.5, forward=forward)
        marks = [strikes[i] for i in bad] or "none"
        # cost of the 70/80/90 butterfly at this forward, for scale
        p = [BlackScholes(forward, K, 0.5).price(v)
             for K, v in zip(strikes, ivs, strict=True)]
        cost = p[0] - 2.0 * p[1] + p[2]
        print(f"F = {forward:6.1f}            violations at strike {marks}"
              f"{'':>{max(0, 12 - len(str(marks)))}}"
              f"  70/80/90 butterfly costs {cost:+.3f}")
    print("the same quotes, different forwards: which butterflies are")
    print("mispriced depends on where the distribution is centred, so the")
    print("forward is a required argument rather than something to infer.")


def calendar() -> None:
    rule("3. total variance that falls with maturity")
    strikes = [80, 90, 100, 110, 120]
    front = [0.42, 0.40, 0.44, 0.41, 0.42]   # 3M, fat with an earnings date
    back = [0.30, 0.26, 0.21, 0.24, 0.29]    # 1Y, calmer
    T_short, T_long = 0.25, 1.00
    bad = calendar_violations(strikes, front, back, T_short, T_long)
    print(f"{'K':>6} {'3M vol':>8} {'1Y vol':>8} {'w(3M)':>9} {'w(1Y)':>9}")
    for i, K in enumerate(strikes):
        w_s, w_l = front[i] ** 2 * T_short, back[i] ** 2 * T_long
        flag = "  <- inverted" if i in bad else ""
        print(f"{K:>6} {front[i]:>8.4f} {back[i]:>8.4f} {w_s:>9.5f}"
              f" {w_l:>9.5f}{flag}")
    print(f"violations             {bad}")
    print("the event vol is concentrated at the money, so only the 100 strike")
    print("inverts; buying the 1Y and selling the 3M there is a free calendar.")


def main() -> None:
    analytic_butterfly()
    discrete_butterfly()
    calendar()


if __name__ == "__main__":
    main()
