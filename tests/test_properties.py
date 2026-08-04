"""Randomised invariant tests.

The rest of the suite pins fixtures: this file pins the statements that have
to hold for *every* admissible input, and checks them on a few thousand
pseudo-random ones. Everything draws from `numpy.random.default_rng(SEED)`
with a fixed seed, so a failure is a failure anyone can reproduce by running
the file.

Each test names the invariant and why it must hold. Where a draw is discarded
the reason is a genuine degeneracy of the quantity being measured (a price
that is flat in sigma to the last bit, a slice with non-positive variance),
never a case that happens to fail.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from volsurf import (
    BlackScholes,
    SABRParams,
    SVIRawParams,
    SVISurface,
    butterfly_violations,
    calendar_violations,
    fit_svi_slice,
    implied_vol,
    sabr_iv,
    svi_density,
    svi_g,
    svi_min_g,
    svi_w,
)

SEED = 20260727


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def _random_bs(rng: np.random.Generator) -> tuple[BlackScholes, float]:
    S = float(rng.uniform(5.0, 500.0))
    return (
        BlackScholes(S=S,
                     K=S * math.exp(float(rng.uniform(-1.2, 1.2))),
                     T=float(rng.uniform(0.02, 5.0)),
                     r=float(rng.uniform(-0.01, 0.08)),
                     q=float(rng.uniform(0.0, 0.05))),
        float(rng.uniform(0.03, 1.2)),
    )


def _random_svi(rng: np.random.Generator, *, b_hi: float = 0.6) -> SVIRawParams:
    return SVIRawParams(a=float(rng.uniform(0.002, 0.20)),
                        b=float(rng.uniform(0.02, b_hi)),
                        rho=float(rng.uniform(-0.90, 0.90)),
                        m=float(rng.uniform(-0.30, 0.30)),
                        sigma=float(rng.uniform(0.05, 0.60)))


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------
def test_put_call_parity_holds_for_every_input():
    """C - P = S e^-qT - K e^-rT.

    Parity is a static replication, not a model result: long a call and short
    a put is a forward, whatever the vol. If it fails, the two branches of
    `price` disagree about the discounting, not about the distribution.
    """
    rng = _rng()
    for _ in range(3000):
        bs, sigma = _random_bs(rng)
        fwd = bs.S * math.exp(-bs.q * bs.T) - bs.K * math.exp(-bs.r * bs.T)
        err = abs(bs.price(sigma) - bs.price(sigma, call=False) - fwd)
        assert err < 1e-12 * bs.S


def test_price_lies_inside_the_static_arbitrage_bounds():
    """max(Se^-qT - Ke^-rT, 0) <= C <= Se^-qT, and the mirror for puts.

    These bounds come from no-arbitrage alone. A price outside them can be
    locked in for a riskless profit, so no parameterisation may produce one.
    """
    rng = _rng(1)
    for _ in range(3000):
        bs, sigma = _random_bs(rng)
        fwd = bs.S * math.exp(-bs.q * bs.T) - bs.K * math.exp(-bs.r * bs.T)
        call, put = bs.price(sigma), bs.price(sigma, call=False)
        assert max(fwd, 0.0) - 1e-12 * bs.S <= call <= bs.S * math.exp(-bs.q * bs.T) + 1e-12 * bs.S
        assert max(-fwd, 0.0) - 1e-12 * bs.S <= put <= bs.K * math.exp(-bs.r * bs.T) + 1e-12 * bs.S


def test_price_is_strictly_increasing_in_volatility():
    """dC/dsigma = vega > 0, so the price orders the vols.

    This is what makes implied vol well defined and a bracketed solver the
    right tool: one price, one vol, and the bracket can never lose the root.
    """
    rng = _rng(2)
    for _ in range(2000):
        bs, sigma = _random_bs(rng)
        if bs.vega(sigma) <= 1e-8 * bs.S:
            continue  # deep wings: the price is flat in sigma to the last bit
        assert bs.price(sigma + 1e-4) > bs.price(sigma)
        assert bs.price(sigma + 1e-4, call=False) > bs.price(sigma, call=False)
        assert bs.vega(sigma) > 0.0


def test_implied_vol_inverts_the_price_it_was_given():
    """implied_vol(price(sigma)) == sigma wherever the price pins sigma.

    The round trip is the only end-to-end statement about the solver that does
    not depend on a reference table. Draws with negligible vega are skipped:
    there the price is genuinely uninformative about the vol, and no solver
    can do better.
    """
    rng = _rng(3)
    checked = 0
    for _ in range(1500):
        bs, sigma = _random_bs(rng)
        if bs.vega(sigma) <= 1e-6 * bs.S:
            continue
        checked += 1
        assert abs(implied_vol(bs.price(sigma), bs) - sigma) < 1e-6
        assert abs(implied_vol(bs.price(sigma, call=False), bs, call=False) - sigma) < 1e-6
    assert checked > 1000


def test_greeks_are_the_derivatives_of_the_price():
    """delta, gamma, vega match central differences of `price`.

    The closed forms are transcribed by hand and are the numbers the rest of
    the portfolio hedges on; differencing the price is an independent route to
    the same quantity. Tolerances are scale-free (S*gamma and vega/S are
    dimensionless) because S ranges over two orders of magnitude here.
    """
    rng = _rng(4)
    for _ in range(1000):
        bs, sigma = _random_bs(rng)
        S, hS, hv = bs.S, bs.S * 1e-4, 1e-4

        def at(spot: float, bs: BlackScholes = bs) -> BlackScholes:
            return BlackScholes(S=spot, K=bs.K, T=bs.T, r=bs.r, q=bs.q)

        up, mid, dn = at(S + hS).price(sigma), bs.price(sigma), at(S - hS).price(sigma)
        assert abs((up - dn) / (2.0 * hS) - bs.delta(sigma)) < 1e-5
        assert abs((up - 2.0 * mid + dn) / (hS * hS) - bs.gamma(sigma)) * S < 1e-4
        fd_vega = (bs.price(sigma + hv) - bs.price(sigma - hv)) / (2.0 * hv)
        assert abs(fd_vega - bs.vega(sigma)) / S < 1e-5


def test_second_strike_derivative_is_the_risk_neutral_density():
    """d2C/dK2 = e^-rT q(K) — Breeden-Litzenberger (1978).

    This is the identity the whole library rests on: it is why a smile is a
    density in disguise and why butterfly arbitrage is a statement about
    curvature. Here the density is the known lognormal one, so both sides are
    available in closed form.
    """
    rng = _rng(5)
    for _ in range(400):
        bs, sigma = _random_bs(rng)
        if bs.T < 0.1 or sigma < 0.1:
            continue  # a needle-sharp density needs a finer difference than h
        K, h = bs.K, bs.K * 1e-4
        d2 = ((math.log(bs.S / K) + (bs.r - bs.q - 0.5 * sigma**2) * bs.T)
              / (sigma * math.sqrt(bs.T)))
        if abs(d2) > 3.0:
            continue  # three sigma out the density is below the price's own ulp

        def at(strike: float, bs: BlackScholes = bs, sigma: float = sigma) -> float:
            return BlackScholes(S=bs.S, K=strike, T=bs.T, r=bs.r, q=bs.q).price(sigma)

        numeric = (at(K + h) - 2.0 * at(K) + at(K - h)) / (h * h)
        analytic = (math.exp(-bs.r * bs.T)
                    * math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
                    / (K * sigma * math.sqrt(bs.T)))
        assert numeric >= -1e-12
        assert abs(numeric - analytic) < 1e-5 * max(analytic, 1e-6)


# --------------------------------------------------------------------------
# SABR
# --------------------------------------------------------------------------
def test_sabr_with_no_vol_of_vol_and_beta_one_is_black():
    """nu = 0, beta = 1 makes SABR the Black model, so sigma_B == alpha exactly.

    Every correction term in Hagan's expansion carries a factor of nu or of
    (1 - beta); switching both off must leave nothing behind. Exact equality
    is the right assertion, not a tolerance.
    """
    rng = _rng(6)
    for _ in range(500):
        alpha = float(rng.uniform(0.02, 1.5))
        p = SABRParams(alpha=alpha, beta=1.0,
                       rho=float(rng.uniform(-0.95, 0.95)), nu=0.0)
        F = float(rng.uniform(10.0, 400.0))
        K = F * math.exp(float(rng.uniform(-1.5, 1.5)))
        assert sabr_iv(F, K, float(rng.uniform(0.01, 8.0)), p) == alpha


def test_sabr_is_invariant_under_rescaling_the_numeraire():
    """sigma_B(lF, lK; alpha l^(1-beta)) = sigma_B(F, K; alpha).

    Implied vol is a pure number, so it cannot depend on whether the forward
    is quoted in cents or in euros — only alpha, which carries units of
    price^(1-beta), may move. This catches any term where a level has crept in
    without its matching power of (FK)^((1-beta)/2).
    """
    rng = _rng(7)
    for _ in range(1000):
        beta = float(rng.uniform(0.0, 1.0))
        alpha = float(rng.uniform(0.05, 0.5))
        rho, nu = float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.01, 1.2))
        F = float(rng.uniform(20.0, 300.0))
        K = F * math.exp(float(rng.uniform(-1.0, 1.0)))
        T = float(rng.uniform(0.05, 3.0))
        lam = float(rng.uniform(0.2, 5.0))
        base = sabr_iv(F, K, T, SABRParams(alpha, beta, rho, nu))
        scaled = sabr_iv(lam * F, lam * K, T,
                         SABRParams(alpha * lam ** (1.0 - beta), beta, rho, nu))
        assert abs(scaled - base) < 1e-10 * base


def test_sabr_at_beta_zero_reflects_strike_and_forward():
    """sigma_B(F, K; rho) = sigma_B(K, F; -rho) when beta = 0.

    Swapping F and K flips the sign of log(F/K) and hence of z, and z/x(z) is
    invariant under (z, rho) -> (-z, -rho). At beta = 0 the time-correction
    term has no rho-odd piece left, so the whole vol is symmetric. (At beta >
    0 the rho*beta*nu*alpha term breaks it — which is why this is asserted at
    beta = 0 and not in general.)
    """
    rng = _rng(8)
    for _ in range(1000):
        alpha = float(rng.uniform(0.5, 5.0))
        rho, nu = float(rng.uniform(-0.9, 0.9)), float(rng.uniform(0.01, 1.2))
        F = float(rng.uniform(20.0, 300.0))
        K = F * math.exp(float(rng.uniform(-1.0, 1.0)))
        T = float(rng.uniform(0.05, 3.0))
        left = sabr_iv(F, K, T, SABRParams(alpha, 0.0, rho, nu))
        right = sabr_iv(K, F, T, SABRParams(alpha, 0.0, -rho, nu))
        assert abs(left - right) < 1e-10 * left


def test_sabr_is_continuous_through_the_money():
    """sigma_B(F, K) -> sigma_B(F, F) linearly in log(F/K), with no step.

    z/x(z) is 0/0 at K = F. Implementations that branch on |K - F| leave a
    band of strikes where the branch has not fired but the logarithm has
    already lost its significant digits, and finite-difference Greeks taken in
    that band are garbage. Walking the strike in by twelve decades, the gap to
    the ATM value must stay proportional to the step the whole way: a dropped
    time-correction factor shows up as a gap of percent size at a step of
    1e-14, which is four orders of magnitude outside this bound.
    """
    rng = _rng(9)
    for _ in range(200):
        p = SABRParams(alpha=float(rng.uniform(0.05, 0.8)),
                       beta=float(rng.uniform(0.0, 1.0)),
                       rho=float(rng.uniform(-0.9, 0.9)),
                       nu=float(rng.uniform(0.05, 1.2)))
        F, T = 100.0, float(rng.uniform(0.05, 3.0))
        atm = sabr_iv(F, F, T, p)
        for e in range(3, 15):
            eps = 10.0**-e
            gap = abs(sabr_iv(F, F * (1.0 + eps), T, p) - atm)
            assert gap < 400.0 * eps * atm + 1e-13


# --------------------------------------------------------------------------
# SVI
# --------------------------------------------------------------------------
def test_svi_closed_form_derivatives_match_finite_differences():
    """w'' > 0 and both derivatives agree with differences of `svi_w`.

    `svi_g` uses hand-derived w' and w''; if either is wrong, g is wrong and
    every arbitrage verdict with it. The sign of w'' matters separately: raw
    SVI is always convex in k, which is precisely why convexity of the smile
    is *not* the butterfly condition.
    """
    rng = _rng(10)
    for _ in range(1000):
        p = _random_svi(rng)
        k = float(rng.uniform(-1.5, 1.5))
        # Different steps for the two differences: the first is limited by
        # truncation, the second by cancellation, and a sharp elbow (small
        # sigma) is unforgiving about both.
        h1, h2 = 1e-6, 1e-4
        w = svi_w(k, p)
        d1 = (svi_w(k + h1, p) - svi_w(k - h1, p)) / (2.0 * h1)
        d2 = (svi_w(k + h2, p) - 2.0 * w + svi_w(k - h2, p)) / (h2 * h2)
        centred = k - p.m
        root = math.hypot(centred, p.sigma)
        assert abs(d1 - p.b * (p.rho + centred / root)) < 1e-7
        assert abs(d2 - p.b * p.sigma**2 / root**3) < 1e-3 * max(d2, 1e-6)
        assert d2 > 0.0


def test_svi_density_and_g_never_disagree_about_the_sign():
    """sign(svi_density(k)) == sign(svi_g(k)).

    The density is g times a strictly positive Gaussian factor, so the two
    must flip together. This is what licenses the library to answer "is this
    slice admissible?" from g alone and never form the density at all.
    """
    rng = _rng(11)
    for _ in range(4000):
        p = SVIRawParams(a=float(rng.uniform(-0.05, 0.20)),
                         b=float(rng.uniform(0.02, 1.0)),
                         rho=float(rng.uniform(-0.95, 0.95)),
                         m=float(rng.uniform(-0.5, 0.5)),
                         sigma=float(rng.uniform(0.05, 0.8)))
        k = float(rng.uniform(-3.0, 3.0))
        try:
            g, dens = svi_g(k, p), svi_density(k, p)
        except ValueError:
            continue  # w(k) <= 0: neither quantity is defined there
        assert (g >= 0.0) == (dens >= 0.0)


def test_svi_density_is_a_unit_mass_that_prices_the_forward():
    """int p(k) dk = 1 and int e^k p(k) dk = 1.

    Both are conditions on the *total* signed mass: the first says the density
    normalises, the second is the martingale condition E[S_T] = F. They hold
    for every raw-SVI slice, arbitrage-free or not — a slice with g < 0
    somewhere still integrates to one, it just does so with a hole in it.
    That is exactly why total mass is useless as an arbitrage screen and g is
    not.
    """
    rng = _rng(12)
    for _ in range(12):
        p = _random_svi(rng, b_hi=0.4)
        R = 4.0
        while R < 400.0 and max(abs(svi_density(R, p)) * math.exp(R),
                                abs(svi_density(-R, p))) > 1e-13:
            R *= 1.5
        n = 1 + 2 * int(300 * R)
        xs = np.linspace(-R, R, n)
        ys = np.array([svi_density(float(k), p) for k in xs])
        weights = np.ones(n)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        h = 2.0 * R / (n - 1) / 3.0
        assert abs(h * float(np.dot(weights, ys)) - 1.0) < 1e-9
        assert abs(h * float(np.dot(weights, ys * np.exp(xs))) - 1.0) < 1e-9


def test_g_far_in_the_wing_is_lees_slope_bound():
    """g(k) -> 1/4 - b^2(1+rho)^2/16 as k -> +inf (and 1-rho on the left).

    Raw SVI's wings are linear with slopes b(1 -/+ rho), and pushing those
    slopes through g leaves only the constant above. It is non-negative iff
    the slope is at most 2, which is Lee's (2004) moment-formula bound on the
    asymptotic slope of total variance. The library's local g-scan therefore
    agrees with the global asymptotic condition rather than contradicting it.
    """
    rng = _rng(13)
    for _ in range(400):
        b = float(rng.uniform(0.02, 1.5))
        rho = float(rng.uniform(-0.95, 0.95))
        p = SVIRawParams(a=0.04, b=b, rho=rho, m=float(rng.uniform(-0.5, 0.5)),
                         sigma=float(rng.uniform(0.05, 0.9)))
        assert abs(svi_g(1e8, p) - (0.25 - (b * (1.0 + rho)) ** 2 / 16.0)) < 1e-6
        assert abs(svi_g(-1e8, p) - (0.25 - (b * (1.0 - rho)) ** 2 / 16.0)) < 1e-6


def test_a_wing_slope_above_two_always_produces_negative_g():
    """b(1 + |rho|) > 2 forces g(k) < 0 somewhere.

    The contrapositive of the wing limit above, and the reason the library can
    be confident a scan over a wide enough range is not missing anything: a
    slice whose wings are too steep cannot hide it.
    """
    rng = _rng(14)
    for _ in range(150):
        rho = float(rng.uniform(-0.9, 0.9))
        b = 2.0 / (1.0 + abs(rho)) * float(rng.uniform(1.05, 3.0))
        p = SVIRawParams(a=float(rng.uniform(0.01, 0.2)), b=b, rho=rho,
                         m=float(rng.uniform(-0.3, 0.3)),
                         sigma=float(rng.uniform(0.05, 0.8)))
        assert svi_min_g(p, -60.0, 60.0, n=3001)[0] < 0.0


def test_min_g_gets_no_larger_as_the_scan_is_refined():
    """svi_min_g on a finer grid is never above the coarse-grid answer.

    A grid scan is an upper bound on the true minimum, so refinement can only
    move it down. The docstring on `svi_butterfly_arbitrage_free` promises
    exactly this and warns that a narrow dip can hide between coarse nodes.
    """
    rng = _rng(15)
    for _ in range(300):
        p = _random_svi(rng)
        coarse = svi_min_g(p, -2.0, 2.0, n=51)[0]
        fine = svi_min_g(p, -2.0, 2.0, n=4001)[0]
        assert fine <= coarse + 1e-12


# --------------------------------------------------------------------------
# calibration and the surface
# --------------------------------------------------------------------------
def test_the_fitter_recovers_the_slice_it_was_given():
    """Exact SVI data in, the same total variance out.

    A least-squares fit to noise-free data generated by the model itself has a
    global minimum at zero. Anything above round-off is the optimiser stopping
    somewhere else — the failure mode the quasi-explicit reduction exists to
    remove, and one that only shows up on parameter draws the fixtures never
    reach.
    """
    rng = _rng(16)
    ks = [float(k) for k in np.linspace(-0.9, 0.9, 25)]
    for _ in range(15):
        p = SVIRawParams(a=float(rng.uniform(0.002, 0.08)),
                         b=float(rng.uniform(0.05, 0.60)),
                         rho=float(rng.uniform(-0.85, 0.40)),
                         m=float(rng.uniform(-0.10, 0.10)),
                         sigma=float(rng.uniform(0.05, 0.50)))
        ws = [svi_w(k, p) for k in ks]
        if min(ws) <= 1e-6:
            continue  # non-positive variance is not a slice anyone quotes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_svi_slice(ks, ws)
        assert max(abs(svi_w(k, fit) / w - 1.0) for k, w in zip(ks, ws, strict=True)) < 1e-5


def test_interpolating_ordered_slices_keeps_them_ordered():
    """If w is non-decreasing in T at the fitted expiries, it is everywhere.

    `SVISurface` interpolates total variance linearly in T, and a convex
    combination of two ordered numbers stays between them. This is the whole
    argument for working in total variance rather than in implied vol, so it
    deserves to be checked off the fitted nodes and not only on them.
    """
    rng = _rng(17)
    for _ in range(60):
        base = _random_svi(rng, b_hi=0.2)
        expiries = [1 / 12, 0.25, 0.5, 1.0, 2.0]
        params = [SVIRawParams(a=base.a * T, b=base.b * T, rho=base.rho,
                               m=base.m, sigma=base.sigma) for T in expiries]
        surf = SVISurface(expiries=list(expiries), params=params)
        assert surf.calendar_arbitrage_free()
        for k in np.linspace(-1.0, 1.0, 21):
            prev = -math.inf
            for T in np.linspace(0.005, 3.0, 200):
                cur = surf.total_variance(float(k), float(T))
                assert cur >= prev - 1e-12
                prev = cur


def test_short_end_implied_vol_stays_bounded():
    """iv(k, T) does not blow up as T -> 0 below the first fitted expiry.

    Holding w flat below the front expiry sends sqrt(w/T) to infinity; scaling
    w proportionally to T holds implied vol at the front slice's level, which
    is the only boundary behaviour that is not an artefact.
    """
    rng = _rng(18)
    for _ in range(50):
        p = _random_svi(rng, b_hi=0.2)
        surf = SVISurface(expiries=[0.25, 1.0],
                          params=[p, SVIRawParams(a=p.a * 4, b=p.b * 4, rho=p.rho,
                                                  m=p.m, sigma=p.sigma)])
        front = surf.iv(0.0, 0.25)
        for T in (1e-6, 1e-4, 1e-2, 0.1):
            assert abs(surf.iv(0.0, T) - front) < 1e-9


# --------------------------------------------------------------------------
# the discrete screens
# --------------------------------------------------------------------------
def test_the_discrete_screen_clears_slices_the_analytic_one_clears():
    """No butterfly violations on quotes generated from a slice with g >= 0.

    The two screens share no code: one differences Black-Scholes prices at a
    forward, the other evaluates a closed-form g(k). Agreement on
    arbitrage-free input is the only cheap way to know that neither is simply
    always saying yes.
    """
    rng = _rng(19)
    checked = 0
    for _ in range(120):
        p = SVIRawParams(a=float(rng.uniform(0.01, 0.06)),
                         b=float(rng.uniform(0.03, 0.25)),
                         rho=float(rng.uniform(-0.8, 0.2)),
                         m=float(rng.uniform(-0.05, 0.05)),
                         sigma=float(rng.uniform(0.10, 0.50)))
        if svi_min_g(p, -0.7, 0.7, n=1201)[0] < 0.0:
            continue
        checked += 1
        T, F = 0.5, 100.0
        ks = [float(k) for k in np.linspace(-0.5, 0.5, 21)]
        strikes = [F * math.exp(k) for k in ks]
        ivs = [math.sqrt(svi_w(k, p) / T) for k in ks]
        assert butterfly_violations(strikes, ivs, T=T, forward=F) == []
    assert checked > 80


def test_calendar_screen_flags_exactly_the_inverted_strikes():
    """The calendar screen returns the strikes where w falls, and only those.

    Total variance is the coordinate in which the calendar condition is a
    plain ordering, so the answer is knowable in advance: build the long-dated
    vols from the short-dated ones by scaling w, push a chosen subset below
    the line, and the returned index set must be that subset.
    """
    rng = _rng(20)
    for _ in range(300):
        n = int(rng.integers(4, 12))
        strikes = [80.0 + 5.0 * i for i in range(n)]
        T_s, T_l = 0.25, 1.0
        ivs_short = [float(rng.uniform(0.10, 0.60)) for _ in range(n)]
        w_short = [v * v * T_s for v in ivs_short]
        broken = {int(i) for i in rng.choice(n, size=int(rng.integers(0, n)),
                                             replace=False)}
        ivs_long = [
            math.sqrt((w * (0.5 if i in broken else 1.5)) / T_l)
            for i, w in enumerate(w_short)
        ]
        found = calendar_violations(strikes, ivs_short, ivs_long, T_s, T_l)
        assert set(found) == broken
