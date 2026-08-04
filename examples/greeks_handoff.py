"""From quotes to a scenario P&L vector — what the surface is actually for.

Run:  PYTHONPATH=. python examples/greeks_handoff.py

`volsurf` is the vol layer of a set of small libraries: it turns a screen of
option quotes into a function sigma(k, T) that other code can evaluate at
strikes and maturities nobody quoted. Two things sit downstream of that
function in this portfolio:

  * `optune` — the pricing engine (Black-Scholes, CRR trees, Monte Carlo,
    exotic payoffs, AAD Greeks). It needs a volatility per contract; it does
    not care where the number came from.
  * `risk` — VaR, expected shortfall, copula loss simulation. It needs a
    vector of P&L under scenarios, not a surface.

This example is the join between the three, written so it runs on its own:
the pricing here is the library's own `BlackScholes` rather than an import of
`optune`, and the loss statistics at the end are three lines rather than an
import of `risk`. Everything upstream — the quote screen — is inlined too.

The order matters. The arbitrage screens run *before* the Greeks, because a
gamma is a second derivative of the same curve whose second derivative is the
risk-neutral density. If the density is negative somewhere, the gamma
reported there is the gamma of something that is not a probability
distribution, and no amount of care downstream repairs that.
"""

from __future__ import annotations

import math

from volsurf import BlackScholes, SVIRawParams, SVISurface, fit_svi_surface, svi_w

# ---------------------------------------------------------------------------
# the quote screen: one name, three listed expiries, nine strikes each
# ---------------------------------------------------------------------------
SPOT = 412.50
RATE = 0.042
DIV = 0.011

STRIKES = [330.0, 360.0, 385.0, 400.0, 412.5, 425.0, 445.0, 470.0, 500.0]

# Implied vols as quoted (mid), by expiry in years. This screen is invented —
# no vendor feed is being redistributed here — but the shape is not: a steep
# short-dated put skew that flattens with maturity, a mild call-wing smirk,
# and a term structure that rises a little at the money.
QUOTES: dict[float, list[float]] = {
    0.0849: [0.3412, 0.2963, 0.2620, 0.2461, 0.2338, 0.2241, 0.2145, 0.2110, 0.2168],
    0.2521: [0.3121, 0.2807, 0.2560, 0.2444, 0.2352, 0.2272, 0.2185, 0.2124, 0.2129],
    0.5041: [0.2946, 0.2717, 0.2529, 0.2440, 0.2367, 0.2301, 0.2224, 0.2158, 0.2131],
}

EXPIRY_LABEL = {0.0849: "31 Aug", 0.2521: "31 Oct", 0.5041: "30 Jan"}


def forward(T: float) -> float:
    """Carry the spot to the expiry — SVI lives in log(K / F), not log(K / S)."""
    return SPOT * math.exp((RATE - DIV) * T)


# ---------------------------------------------------------------------------
# the book: five positions, one of them at an expiry nobody lists
# ---------------------------------------------------------------------------
# (label, strike, expiry in years, call?, contracts) — 100 shares per contract
BOOK = [
    ("short Aug 445 calls", 445.0, 0.0849, True, -250),
    ("long Aug 385 puts", 385.0, 0.0849, False, 400),
    ("long Oct straddle, call leg", 412.5, 0.2521, True, 120),
    ("long Oct straddle, put leg", 412.5, 0.2521, False, 120),
    ("short Dec 470 calls", 470.0, 0.3699, True, -300),
]
CONTRACT_SIZE = 100


def fit_surface() -> SVISurface:
    slices = []
    for T, ivs in QUOTES.items():
        F = forward(T)
        ks = [math.log(K / F) for K in STRIKES]
        slices.append((T, ks, [iv * iv * T for iv in ivs]))
    return fit_svi_surface(slices)


def screen(surf: SVISurface) -> None:
    print("arbitrage screens on the fitted surface")
    k_lo = math.log(min(STRIKES) / forward(max(QUOTES))) - 0.05
    k_hi = math.log(max(STRIKES) / forward(min(QUOTES))) + 0.05
    ks = [k_lo + (k_hi - k_lo) * i / 40.0 for i in range(41)]
    print(f"  calendar  (w non-decreasing in T): {surf.calendar_arbitrage_free(ks)}")
    print(f"  butterfly (g(k) >= 0 per slice):   "
          f"{surf.butterfly_arbitrage_free(k_lo, k_hi, n=801)}")
    print(f"  scanned k in [{k_lo:+.3f}, {k_hi:+.3f}]  "
          f"(K/F from {math.exp(k_lo):.2f} to {math.exp(k_hi):.2f})")


def fit_quality(surf: SVISurface) -> None:
    print("\nfit residuals against the quotes, in vol points")
    print("  expiry     max err   rms err   ATM vol")
    for T, ivs in QUOTES.items():
        F = forward(T)
        errs = [surf.iv(math.log(K / F), T) - iv for K, iv in zip(STRIKES, ivs, strict=True)]
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"  {EXPIRY_LABEL[T]:9s}  {100 * max(abs(e) for e in errs):7.4f}  "
              f"{100 * rms:7.4f}   {100 * surf.iv(0.0, T):6.2f}%")


# ---------------------------------------------------------------------------
# valuation: read a vol off the surface, then price and differentiate
# ---------------------------------------------------------------------------
def value_book(surf: SVISurface, spot: float) -> tuple[float, float, float, float]:
    """(value, delta, gamma, vega) of the whole book, in currency units.

    The surface is parameterised in log-moneyness, so moving the spot slides
    each strike along the smile rather than dragging the smile with it: this
    is a sticky-delta book. That is a choice, not a fact about the market, and
    it is the reason the gamma below is not the Black-Scholes gamma at a fixed
    vol. Say which convention you used or the number means nothing.
    """
    value = delta = gamma = vega = 0.0
    for _, K, T, call, qty in BOOK:
        F = spot * math.exp((RATE - DIV) * T)
        sigma = surf.iv(math.log(K / F), T)
        bs = BlackScholes(S=spot, K=K, T=T, r=RATE, q=DIV)
        scale = qty * CONTRACT_SIZE
        value += scale * bs.price(sigma, call=call)
        delta += scale * bs.delta(sigma, call=call)
        gamma += scale * bs.gamma(sigma)
        vega += scale * bs.vega(sigma) / 100.0  # per vol point
    return value, delta, gamma, vega


def position_report(surf: SVISurface) -> None:
    print("\nthe book, valued off the surface")
    print("  position                     K       T    src    vol     value")
    for label, K, T, call, qty in BOOK:
        F = forward(T)
        sigma = surf.iv(math.log(K / F), T)
        src = "quoted" if T in QUOTES else "interp"
        bs = BlackScholes(S=SPOT, K=K, T=T, r=RATE, q=DIV)
        val = qty * CONTRACT_SIZE * bs.price(sigma, call=call)
        kind = "C" if call else "P"
        print(f"  {label:26s} {K:6.1f}{kind}  {T:.4f}  {src}  {100 * sigma:6.2f}%  "
              f"{val:+11,.0f}")
    value, delta, gamma, vega = value_book(surf, SPOT)
    print(f"  {'book':26s} {'':7s}  {'':6s}  {'':6s} {'':7s}  {value:+11,.0f}")
    print(f"\n  delta {delta:+,.0f} shares    gamma {gamma:+.2f} shares per $1    "
          f"vega {vega:+,.0f} per vol point")
    print("  (delta, gamma, vega are the numbers an AAD pricer such as `optune` "
          "would\n   produce from these same vols; here they come from the "
          "closed forms in\n   volsurf.black_scholes, which is enough to show the "
          "handoff.)")


# ---------------------------------------------------------------------------
# scenarios: shock the spot and the surface, revalue, hand off a P&L vector
# ---------------------------------------------------------------------------
def shocked(surf: SVISurface, vol_mult: float, skew_shift: float) -> SVISurface:
    """A surface with total variance scaled and the skew steepened.

    Scaling w by `vol_mult**2` is a proportional vol shock at every strike;
    pushing rho down steepens the put wing. Both act on the SVI parameters
    directly, so the shocked surface is still an SVI surface and can be run
    back through the same arbitrage screens — which is the point of shocking
    the parameters rather than the vols.
    """
    lam = vol_mult * vol_mult
    params = [SVIRawParams(a=p.a * lam, b=p.b * lam,
                           rho=max(-0.99, p.rho - skew_shift),
                           m=p.m, sigma=p.sigma)
              for p in surf.params]
    return SVISurface(expiries=list(surf.expiries), params=params)


SCENARIOS = [
    ("base", 0.00, 1.00, 0.00),
    ("spot -10%, vol +35%", -0.10, 1.35, 0.06),
    ("spot -5%, vol +15%", -0.05, 1.15, 0.03),
    ("spot -2%, vol +5%", -0.02, 1.05, 0.01),
    ("spot flat, vol -8%", 0.00, 0.92, -0.01),
    ("spot +2%, vol -6%", 0.02, 0.94, -0.01),
    ("spot +5%, vol -12%", 0.05, 0.88, -0.02),
    ("spot +10%, vol -18%", 0.10, 0.82, -0.03),
]


def scenario_table(surf: SVISurface) -> list[float]:
    base_value, _, _, _ = value_book(surf, SPOT)
    print("\nscenario P&L (spot and surface shocked together)")
    print("  scenario                 spot    ATM 3M vol    P&L      arb-free")
    pnl = []
    for label, ds, vol_mult, skew in SCENARIOS:
        shocked_surf = shocked(surf, vol_mult, skew)
        spot = SPOT * (1.0 + ds)
        value, _, _, _ = value_book(shocked_surf, spot)
        pnl.append(value - base_value)
        ok = (shocked_surf.calendar_arbitrage_free()
              and shocked_surf.butterfly_arbitrage_free(-0.5, 0.4, n=401))
        print(f"  {label:22s} {spot:7.2f}     {100 * shocked_surf.iv(0.0, 0.2521):6.2f}%  "
              f"{value - base_value:+10,.0f}    {ok}")
    return pnl


def main() -> None:
    surf = fit_surface()
    print(f"spot {SPOT:.2f}   r {RATE:.3f}   q {DIV:.3f}   "
          f"{len(QUOTES)} expiries x {len(STRIKES)} strikes\n")
    for T in surf.expiries:
        p = surf.params[surf.expiries.index(T)]
        print(f"  {EXPIRY_LABEL[T]}  T={T:.4f}  a={p.a:+.5f} b={p.b:.4f} "
              f"rho={p.rho:+.4f} m={p.m:+.4f} sigma={p.sigma:.4f}")
    print()
    screen(surf)
    fit_quality(surf)
    position_report(surf)

    # A vol nobody quoted: the Dec expiry in the book sits between the Oct and
    # Jan slices, which is the only reason the book can be marked at all.
    T = 0.3699
    F = forward(T)
    print(f"\nDec 470 call: k = {math.log(470.0 / F):+.4f}, T = {T}, "
          f"between {EXPIRY_LABEL[0.2521]} and {EXPIRY_LABEL[0.5041]}")
    print(f"  w interpolated linearly in T: {surf.total_variance(math.log(470.0 / F), T):.6f}"
          f"   -> vol {100 * surf.iv(math.log(470.0 / F), T):.2f}%")
    lo = svi_w(math.log(470.0 / F), surf.params[1])
    hi = svi_w(math.log(470.0 / F), surf.params[2])
    print(f"  bracketed by the two fitted slices: {lo:.6f} <= w <= {hi:.6f}")

    pnl = scenario_table(surf)
    losses = sorted(p for p in pnl)
    print("\nwhat leaves this library")
    print(f"  P&L vector, {len(pnl)} scenarios: "
          f"[{', '.join(f'{p:+,.0f}' for p in pnl)}]")
    print(f"  worst {losses[0]:+,.0f}   mean {sum(pnl) / len(pnl):+,.0f}   "
          f"mean of worst two {sum(losses[:2]) / 2:+,.0f}")
    print("  A risk engine such as `risk` takes it from here: this is a hand-built")
    print("  set of eight scenarios, and eight is not a distribution. The surface's")
    print("  job ended at the vol column.")


if __name__ == "__main__":
    main()
