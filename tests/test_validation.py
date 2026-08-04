"""The claims in `docs/validation.md`, asserted.

`examples/validate.py` is the script that produces the numbers in that
document; this file imports the same reference implementations and asserts the
comparisons, at smaller sample sizes so the suite stays quick. If a number in
the doc drifts, one of these fails.

The reference side of every comparison is deliberately something other than
the code under test: a closed form transcribed from the source paper, a limit
the model must collapse to, or a Monte-Carlo simulation of the SDE that the
formula only approximates.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

from validate import (  # noqa: E402
    VOGT,
    density_moments,
    hagan_atm,
    lognormal_density_in_strike,
    sabr_conditional_mc,
    ssvi_as_raw,
)

from volsurf import (  # noqa: E402
    BlackScholes,
    SABRParams,
    SVIRawParams,
    butterfly_violations,
    implied_vol,
    sabr_iv,
    svi_density,
    svi_g,
    svi_min_g,
    svi_w,
)

SEED = 20260727


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------
def test_zero_vol_price_is_the_discounted_forward_intrinsic():
    """As sigma -> 0 the call is worth max(F - K, 0) discounted, nothing else."""
    bs = BlackScholes(S=100.0, K=90.0, T=1.0, r=0.03, q=0.01)
    intrinsic = max(100.0 * math.exp(-0.01) - 90.0 * math.exp(-0.03), 0.0)
    assert abs(bs.price(1e-12) - intrinsic) < 1e-12
    # and as sigma -> infinity the call converges to the discounted spot
    assert abs(bs.price(60.0) - 100.0 * math.exp(-0.01)) < 1e-9


def test_second_strike_derivative_matches_the_lognormal_density():
    """Breeden-Litzenberger (1978): d2C/dK2 = e^-rT q(K), q the lognormal pdf."""
    S, K, T, r, q, sigma = 100.0, 105.0, 0.75, 0.03, 0.01, 0.28
    h = 1e-3

    def price_at(strike: float) -> float:
        return BlackScholes(S=S, K=strike, T=T, r=r, q=q).price(sigma)

    numeric = (price_at(K + h) - 2.0 * price_at(K) + price_at(K - h)) / (h * h)
    analytic = lognormal_density_in_strike(K, BlackScholes(S=S, K=K, T=T, r=r, q=q), sigma)
    assert abs(numeric / analytic - 1.0) < 1e-6


# --------------------------------------------------------------------------
# SABR against closed forms
# --------------------------------------------------------------------------
def test_sabr_atm_equals_the_hagan_closed_form():
    """sabr_iv(F, F) reproduces Hagan et al. (2002) sigma_B(f, f).

    `hagan_atm` writes the paper's at-the-money case out directly, including
    the (1 + [...] T) factor. `sabr_iv` reaches the same point as the K -> F
    limit of the general formula, through the z/x(z) series. Two different
    code paths, one published expression.
    """
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(1000):
        F = float(rng.uniform(10.0, 300.0))
        T = float(rng.uniform(0.02, 5.0))
        p = SABRParams(alpha=float(rng.uniform(0.05, 3.0)),
                       beta=float(rng.uniform(0.0, 1.0)),
                       rho=float(rng.uniform(-0.95, 0.95)),
                       nu=float(rng.uniform(0.0, 1.5)))
        ref = hagan_atm(F, T, p)
        worst = max(worst, abs(sabr_iv(F, F, T, p) - ref) / ref)
    assert worst < 1e-13


def test_sabr_at_beta_zero_is_the_log_ratio_normal_vol():
    """With beta = 0 and nu -> 0 the Black vol is alpha log(F/K)/(F - K).

    The bracket (FK)^((1-beta)/2)[1 + ln^2/24 + ln^4/1920] in Hagan's formula
    is the fourth-order expansion of (F - K)/log(F/K), so a normal model with
    no vol-of-vol has this exact Black vol. The residual is the O(ln^6)
    truncation and nothing else.
    """
    p = SABRParams(alpha=2.0, beta=0.0, rho=0.0, nu=1e-12)
    for K in (80.0, 90.0, 110.0, 120.0):
        ours = sabr_iv(100.0, K, 1e-8, p)
        ref = 2.0 * math.log(100.0 / K) / (100.0 - K)
        assert abs(ours / ref - 1.0) < 1e-9


# --------------------------------------------------------------------------
# SABR against the SDE
# --------------------------------------------------------------------------
def test_hagan_vols_match_a_monte_carlo_of_the_sabr_sde():
    """At nu^2 T = 0.16 the expansion is within a basis point or two of truth.

    `sabr_conditional_mc` simulates the lognormal SABR SDE and prices calls by
    conditioning on the volatility path, so the reference has nothing to do
    with Hagan's expansion. Tolerance is three Monte-Carlo standard errors
    plus 2 bp of genuine expansion error near the money and 15 bp in the
    wings; `docs/validation.md` carries the measured numbers.
    """
    F0, T = 100.0, 1.0
    p = SABRParams(alpha=0.20, beta=1.0, rho=-0.30, nu=0.40)
    strikes = (70.0, 85.0, 100.0, 115.0, 130.0)
    res = sabr_conditional_mc(F0, T, p, strikes, n_pairs=25_000, n_steps=150,
                              seed=SEED)
    for K in strikes:
        price, se = res[K]
        bs = BlackScholes(S=F0, K=K, T=T)
        mc_iv = implied_vol(price, bs)
        se_iv = implied_vol(price + se, bs) - mc_iv
        budget = 3.0 * se_iv + (15e-4 if abs(math.log(K / F0)) > 0.2 else 2e-4)
        assert abs(sabr_iv(F0, K, T, p) - mc_iv) < budget


def test_the_expansion_visibly_breaks_down_at_large_vol_of_vol():
    """At nu^2 T = 3.2 Hagan is more than a vol point wide of the true price.

    Asserted rather than hidden: the formula is a first-order expansion in
    nu^2 T and this is what its failure looks like. A change that made this
    test pass with a small tolerance would mean `sabr_iv` had stopped being
    Hagan's formula.
    """
    F0, T = 100.0, 5.0
    p = SABRParams(alpha=0.20, beta=1.0, rho=-0.30, nu=0.80)
    res = sabr_conditional_mc(F0, T, p, (50.0, 100.0), n_pairs=25_000,
                              n_steps=150, seed=SEED)
    gaps = {}
    for K in (50.0, 100.0):
        price, _ = res[K]
        mc_iv = implied_vol(price, BlackScholes(S=F0, K=K, T=T))
        gaps[K] = sabr_iv(F0, K, T, p) - mc_iv
    assert gaps[100.0] > 0.02          # Hagan too high by 2+ vol points ATM
    assert gaps[50.0] > 0.08           # and by 8+ in the left wing


# --------------------------------------------------------------------------
# SVI against Gatheral-Jacquier and Lee
# --------------------------------------------------------------------------
def test_a_flat_slice_reproduces_the_black_scholes_density():
    """b = 0 makes w constant, and the SVI density becomes the lognormal one."""
    sigma, T = 0.25, 1.0
    flat = SVIRawParams(a=sigma * sigma * T, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    for k in np.linspace(-2.0, 2.0, 201):
        w = sigma * sigma * T
        d_minus = -float(k) / math.sqrt(w) - 0.5 * math.sqrt(w)
        ref = math.exp(-0.5 * d_minus * d_minus) / math.sqrt(2.0 * math.pi * w)
        assert abs(svi_density(float(k), flat) - ref) < 1e-15


def test_the_implied_density_is_a_unit_mass_even_when_it_goes_negative():
    """int p dk = 1 and int e^k p dk = 1 for the Vogt slice too.

    The Vogt slice admits butterfly arbitrage, yet its signed density still
    normalises and still prices the forward. Mass is not the test; sign is.
    """
    for p in (SVIRawParams(a=0.012, b=0.10, rho=-0.60, m=0.02, sigma=0.18), VOGT):
        mass, martingale = density_moments(p)
        assert abs(mass - 1.0) < 1e-10
        assert abs(martingale - 1.0) < 1e-10
    assert svi_min_g(VOGT, -1.5, 1.5, n=3001)[0] < 0.0


def test_the_wing_limit_of_g_is_lees_bound():
    """g(+inf) = 1/4 - b^2(1+rho)^2/16, which is >= 0 exactly when the wing
    slope b(1+rho) is at most 2 — Lee's (2004) moment-formula bound."""
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        b = float(rng.uniform(0.02, 1.5))
        rho = float(rng.uniform(-0.95, 0.95))
        p = SVIRawParams(a=0.04, b=b, rho=rho, m=0.0, sigma=0.3)
        assert abs(svi_g(1e8, p) - (0.25 - (b * (1.0 + rho)) ** 2 / 16.0)) < 1e-6


def test_the_gatheral_jacquier_ssvi_conditions_imply_non_negative_g():
    """SSVI slices inside the published sufficient conditions never dip.

    Gatheral and Jacquier (2014) give, for an SSVI surface with parameters
    (theta, phi, rho), the sufficient conditions

        theta phi (1 + |rho|) < 4  and  theta phi^2 (1 + |rho|) <= 4

    for freedom from butterfly arbitrage. `ssvi_as_raw` maps such a slice to
    raw SVI, and the library's own g-scan must then find nothing. Draws are
    taken at 90-99.9% of the first bound so the test lives near the boundary
    rather than deep inside it.
    """
    rng = np.random.default_rng(SEED + 2)
    drawn = 0
    while drawn < 150:
        rho = float(rng.uniform(-0.98, 0.98))
        s = 1.0 + abs(rho)
        phi = float(rng.uniform(0.1, 8.0))
        theta = (4.0 / s) * float(rng.uniform(0.90, 0.999)) / phi
        if theta * phi * phi * s > 4.0:
            continue
        drawn += 1
        assert svi_min_g(ssvi_as_raw(theta, phi, rho), -40.0, 40.0, n=6001)[0] >= 0.0


def test_both_butterfly_screens_locate_the_same_bad_strikes():
    """The analytic g(k) < 0 region and the discrete price screen agree.

    On the Vogt slice one screen evaluates a closed form and the other
    differences Black-Scholes prices at a forward. They share no code, so
    agreement to within the quote spacing is real evidence.
    """
    ks = [float(k) for k in np.linspace(0.40, 1.40, 21)]
    strikes = [100.0 * math.exp(k) for k in ks]
    ivs = [math.sqrt(svi_w(k, VOGT)) for k in ks]
    flagged = [ks[i] for i in butterfly_violations(strikes, ivs, T=1.0,
                                                   forward=100.0, eps=1e-9)]
    negative = [float(k) for k in np.linspace(0.40, 1.40, 2001)
                if svi_g(float(k), VOGT) < 0.0]
    spacing = ks[1] - ks[0]
    assert abs(min(flagged) - min(negative)) < spacing
    assert abs(max(flagged) - max(negative)) < spacing
