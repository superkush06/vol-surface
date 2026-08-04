"""Check `volsurf` against things that are not `volsurf`.

Run:  PYTHONPATH=. python examples/validate.py

Every number in `docs/validation.md` is printed by this script. The point of
each section is that the right-hand side comes from somewhere outside the
library: a closed form stated in the source paper, a limit the model must
collapse to, an independent Monte-Carlo simulation of the SDE the formula is
an approximation *of*, or a second screen implemented on a different
principle. Where the two sides disagree, the disagreement is printed with the
rest — see the notes in `docs/validation.md`.

numpy is used for the Monte-Carlo path generation and the quadrature only;
`volsurf` itself stays standard-library.
"""

from __future__ import annotations

import math

import numpy as np

from volsurf import (
    BlackScholes,
    SABRParams,
    SVIRawParams,
    butterfly_violations,
    fit_svi_surface,
    implied_vol,
    sabr_iv,
    svi_density,
    svi_g,
    svi_min_g,
    svi_w,
)

SEED = 20260727

# Axel Vogt's slice, the standard SVI example that admits butterfly arbitrage.
VOGT = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------------------
# reference implementations — deliberately not the ones under test
# ---------------------------------------------------------------------------
def hagan_atm(F: float, T: float, p: SABRParams) -> float:
    """Hagan et al. (2002) at-the-money vol sigma_B(f, f), written out directly.

    This is the K = F case of their implied-vol expansion, transcribed from the
    paper rather than obtained by taking a limit of `sabr_iv`.
    """
    om = 1.0 - p.beta
    return p.alpha / F**om * (
        1.0
        + (om * om * p.alpha**2 / (24.0 * F ** (2.0 * om))
           + 0.25 * p.rho * p.beta * p.nu * p.alpha / F**om
           + (2.0 - 3.0 * p.rho**2) * p.nu**2 / 24.0) * T
    )


def sabr_iv_obloj(F: float, K: float, T: float, p: SABRParams) -> float:
    """Hagan's expansion with Obloj's (2008) zeta in place of z.

    Obloj notes that Hagan et al.'s

        z = (nu/alpha) (FK)^((1-beta)/2) log(F/K)

    is itself an approximation to

        zeta = (nu/alpha) (F^(1-beta) - K^(1-beta)) / (1-beta),

    the two agreeing to leading order in (1-beta) log(F/K). Everything else is
    unchanged. `volsurf` ships the original z; this is here to measure how much
    that choice costs.
    """
    alpha, beta, rho, nu = p.alpha, p.beta, p.rho, p.nu
    om = 1.0 - beta
    log_fk = math.log(F / K)
    fk_beta = (F * K) ** (om / 2.0)
    A = alpha / (fk_beta * (1.0 + om**2 * log_fk**2 / 24.0
                            + om**4 * log_fk**4 / 1920.0))
    z = (nu / alpha) * (F**om - K**om) / om if om > 0.0 else (nu / alpha) * log_fk
    if abs(z) < 1e-6:
        ratio = 1.0 + 0.5 * rho * z + (3.0 * rho * rho - 2.0) * z * z / 12.0
    else:
        x_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        ratio = z / x_z
    corr = 1.0 + ((om * alpha) ** 2 / (24.0 * fk_beta**2)
                  + 0.25 * rho * beta * nu * alpha / fk_beta
                  + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0) * T
    return A * ratio * corr


def lognormal_density_in_strike(K: float, bs: BlackScholes, sigma: float) -> float:
    """The Black-Scholes risk-neutral density, d^2C/dK^2 = e^{-rT} q(K)."""
    d2 = ((math.log(bs.S / K) + (bs.r - bs.q - 0.5 * sigma**2) * bs.T)
          / (sigma * math.sqrt(bs.T)))
    pdf = math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
    return math.exp(-bs.r * bs.T) * pdf / (K * sigma * math.sqrt(bs.T))


_erf = np.vectorize(math.erf, otypes=[float])


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def sabr_conditional_mc(F0: float, T: float, p: SABRParams, strikes,
                        *, n_pairs: int = 100_000, n_steps: int = 500,
                        seed: int = SEED, chunk: int = 25_000):
    """Monte-Carlo call prices under the *exact* lognormal SABR SDE (beta = 1).

    Returns {K: (price, standard_error)} for zero rates, so the price is a
    forward price and the strike is a forward strike.

    With beta = 1 the model is

        dF = alpha F dW1,   d(alpha) = nu alpha dW2,   d<W1,W2> = rho dt.

    Split dW1 = rho dW2 + sqrt(1-rho^2) dW_perp. Conditional on the whole
    alpha path, log F_T is Gaussian, so the call price is a Black-Scholes
    price at

        F_eff = F0 exp(rho (alpha_T - alpha_0)/nu - rho^2 V / 2),
        sigma_eff = sqrt((1 - rho^2) V / T),        V = int_0^T alpha_t^2 dt,

    using int alpha dW2 = (alpha_T - alpha_0)/nu, which holds because alpha is
    a driftless geometric Brownian motion. Conditioning on the volatility path
    like this is the Romano-Touzi (1997) mixing argument; it removes the
    variance contributed by W_perp entirely, so a few hundred thousand paths
    price the wings to a fraction of a basis point of vol.

    The alpha path is exact (lognormal increments); the only discretisation is
    the trapezoid rule for V, and the printed step-refinement check shows it is
    not what limits the comparison. Antithetic pairs on the driving normals.
    """
    a0, rho, nu = p.alpha, p.rho, p.nu
    if p.beta != 1.0:
        raise ValueError("the conditioning argument used here needs beta = 1")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    root_dt, root_T = math.sqrt(dt), math.sqrt(T)
    acc = {K: [0.0, 0.0, 0] for K in strikes}
    done = 0
    while done < n_pairs:
        n = min(chunk, n_pairs - done)
        done += n
        z = rng.standard_normal((n, n_steps))
        z = np.concatenate([z, -z], axis=0)
        alpha = np.full(z.shape[0], a0)
        V = np.zeros(z.shape[0])
        for j in range(n_steps):
            nxt = alpha * np.exp(nu * root_dt * z[:, j] - 0.5 * nu * nu * dt)
            V += 0.5 * dt * (alpha * alpha + nxt * nxt)
            alpha = nxt
        f_eff = F0 * np.exp(rho * (alpha - a0) / nu - 0.5 * rho * rho * V)
        s_eff = np.sqrt((1.0 - rho * rho) * V / T)
        for K in strikes:
            d1 = (np.log(f_eff / K) + 0.5 * s_eff * s_eff * T) / (s_eff * root_T)
            c = f_eff * _norm_cdf(d1) - K * _norm_cdf(d1 - s_eff * root_T)
            acc[K][0] += float(c.sum())
            acc[K][1] += float((c * c).sum())
            acc[K][2] += len(c)
    out = {}
    for K in strikes:
        s, s2, n = acc[K]
        mean = s / n
        out[K] = (mean, math.sqrt(max(s2 / n - mean * mean, 0.0) / n))
    return out


def simpson(f, lo: float, hi: float, n: int) -> float:
    """Composite Simpson on n (odd) nodes."""
    x = np.linspace(lo, hi, n)
    y = np.array([f(float(t)) for t in x])
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((hi - lo) / (n - 1) / 3.0 * np.dot(w, y))


def density_moments(p: SVIRawParams, *, tail: float = 1e-13):
    """(total mass, E[e^k]) of the SVI implied density, integrated to the tails.

    The range is widened until both the density and its e^k-weighted version
    are below `tail` at the endpoints, so the answer is not a statement about
    where the integration was truncated.
    """
    R = 4.0
    while R < 400.0 and max(abs(svi_density(R, p)) * math.exp(R),
                            abs(svi_density(-R, p))) > tail:
        R *= 1.5
    n = 1 + 2 * int(400 * R)
    mass = simpson(lambda k: svi_density(k, p), -R, R, n)
    mart = simpson(lambda k: math.exp(k) * svi_density(k, p), -R, R, n)
    return mass, mart


def ssvi_as_raw(theta: float, phi: float, rho: float) -> SVIRawParams:
    """The raw-SVI parameters of one SSVI slice.

        w(k) = theta/2 {1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)}

    matches raw SVI with b = theta phi / 2, m = -rho/phi,
    sigma = sqrt(1-rho^2)/phi and a = theta (1-rho^2)/2 = b sigma sqrt(1-rho^2).
    """
    return SVIRawParams(a=theta * (1.0 - rho * rho) / 2.0,
                        b=theta * phi / 2.0,
                        rho=rho,
                        m=-rho / phi,
                        sigma=math.sqrt(1.0 - rho * rho) / phi)


# ---------------------------------------------------------------------------
# 1. Black-Scholes: parity, static bounds, limits, Breeden-Litzenberger
# ---------------------------------------------------------------------------
def black_scholes_section() -> None:
    head("1. Black-Scholes against its own identities")
    rng = np.random.default_rng(SEED)
    worst_parity = 0.0
    worst_bounds = 0.0
    worst_rt = 0.0
    for _ in range(5000):
        S = float(rng.uniform(5.0, 500.0))
        K = S * math.exp(float(rng.uniform(-1.5, 1.5)))
        T = float(rng.uniform(0.01, 5.0))
        r = float(rng.uniform(-0.01, 0.08))
        q = float(rng.uniform(0.0, 0.05))
        sig = float(rng.uniform(0.02, 1.5))
        bs = BlackScholes(S=S, K=K, T=T, r=r, q=q)
        call, put = bs.price(sig), bs.price(sig, call=False)
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        worst_parity = max(worst_parity, abs(call - put - fwd) / S)
        worst_bounds = max(worst_bounds,
                           max(0.0, max(fwd, 0.0) - call, call - S * math.exp(-q * T)) / S)
        # Only round-trip where the price actually pins the vol: far enough out
        # of the money the price is flat in sigma to the last bit, and no
        # solver can recover it.
        if bs.vega(sig) > 1e-6 * S:
            worst_rt = max(worst_rt, abs(implied_vol(call, bs) - sig))
    print(f"put-call parity   max |C - P - (Se^-qT - Ke^-rT)| / S = {worst_parity:.3e}")
    print(f"static bounds     max violation of max(F-K,0) <= C <= Se^-qT / S = {worst_bounds:.3e}")
    print(f"IV round trip     max |implied_vol(price(sigma)) - sigma| = {worst_rt:.3e}")

    bs = BlackScholes(S=100.0, K=90.0, T=1.0, r=0.03, q=0.01)
    zero_vol = bs.price(1e-12)
    disc_intrinsic = max(100.0 * math.exp(-0.01) - 90.0 * math.exp(-0.03), 0.0)
    print(f"sigma -> 0        C = {zero_vol!r}   discounted intrinsic = {disc_intrinsic!r}")
    print(f"sigma -> inf      C = {bs.price(60.0)!r}   Se^-qT = {100.0 * math.exp(-0.01)!r}")

    S, K, T, r, q, sig = 100.0, 105.0, 0.75, 0.03, 0.01, 0.28
    h = 1e-3

    def price_at(strike: float) -> float:
        return BlackScholes(S=S, K=strike, T=T, r=r, q=q).price(sig)

    numeric = (price_at(K + h) - 2.0 * price_at(K) + price_at(K - h)) / (h * h)
    analytic = lognormal_density_in_strike(K, BlackScholes(S=S, K=K, T=T, r=r, q=q), sig)
    print(f"Breeden-Litzenberger at K=105:  d2C/dK2 = {numeric:.12f}   "
          f"e^-rT q(K) = {analytic:.12f}   rel = {abs(numeric / analytic - 1.0):.2e}")


# ---------------------------------------------------------------------------
# 2. SABR against closed forms and against the SDE
# ---------------------------------------------------------------------------
def sabr_limits_section() -> None:
    head("2. SABR against the limits the model must collapse to")
    p = SABRParams(alpha=0.28, beta=1.0, rho=-0.70, nu=0.0)
    worst = max(abs(sabr_iv(100.0, K, T, p) - 0.28)
                for K in (40.0, 70.0, 100.0, 130.0, 200.0, 400.0)
                for T in (0.1, 1.0, 5.0))
    print(f"beta=1, nu=0 is Black:   max |sabr_iv - alpha| over 18 (K,T) = {worst!r}")

    rng = np.random.default_rng(SEED)
    worst_atm = 0.0
    for _ in range(2000):
        F = float(rng.uniform(10.0, 300.0))
        T = float(rng.uniform(0.02, 5.0))
        pp = SABRParams(alpha=float(rng.uniform(0.05, 3.0)),
                        beta=float(rng.uniform(0.0, 1.0)),
                        rho=float(rng.uniform(-0.95, 0.95)),
                        nu=float(rng.uniform(0.0, 1.5)))
        ref = hagan_atm(F, T, pp)
        worst_atm = max(worst_atm, abs(sabr_iv(F, F, T, pp) - ref) / ref)
    print(f"ATM = Hagan sigma_B(f,f): max rel err over 2000 random parameter "
          f"sets = {worst_atm:.3e}")

    # beta = 0, nu -> 0: the Hagan prefactor is the 4th-order expansion of
    # (F-K)/log(F/K), so the Black vol must approach alpha log(F/K)/(F-K).
    p0 = SABRParams(alpha=2.0, beta=0.0, rho=0.0, nu=1e-12)
    print("beta=0, nu->0 vs alpha*log(F/K)/(F-K):")
    for K in (80.0, 90.0, 110.0, 120.0):
        ours = sabr_iv(100.0, K, 1e-8, p0)
        ref = 2.0 * math.log(100.0 / K) / (100.0 - K)
        print(f"   K={K:6.1f}   ours {ours:.12f}   ref {ref:.12f}   "
              f"rel {abs(ours / ref - 1.0):.2e}")

    # A documented refinement we do not implement: Obloj's zeta.
    p_half = SABRParams(alpha=2.0, beta=0.5, rho=-0.30, nu=0.40)
    print("\nHagan's z vs Obloj's zeta at beta=0.5, alpha=2, rho=-0.3, nu=0.4, "
          "F=100, T=1:")
    print("      K    ships (z)   Obloj (zeta)   difference (bp of vol)")
    for K in (50.0, 70.0, 100.0, 140.0, 200.0):
        ours = sabr_iv(100.0, K, 1.0, p_half)
        alt = sabr_iv_obloj(100.0, K, 1.0, p_half)
        print(f"   {K:6.1f}   {ours:.6f}     {alt:.6f}       {1e4 * (ours - alt):+8.2f}")


def sabr_mc_section() -> None:
    head("3. SABR against a Monte-Carlo of the SDE it approximates")
    for label, T, nu, strikes in (
        ("nu^2 T = 0.16", 1.0, 0.40, (70.0, 85.0, 100.0, 115.0, 130.0)),
        ("nu^2 T = 0.72", 2.0, 0.60, (60.0, 80.0, 100.0, 125.0, 160.0)),
        ("nu^2 T = 3.20", 5.0, 0.80, (50.0, 75.0, 100.0, 150.0, 200.0)),
    ):
        p = SABRParams(alpha=0.20, beta=1.0, rho=-0.30, nu=nu)
        res = sabr_conditional_mc(100.0, T, p, strikes)
        print(f"\nalpha=0.20 beta=1 rho=-0.30 nu={nu:.2f} T={T:.1f}   ({label})")
        print("     K    MC vol     Hagan vol   Hagan - MC (bp)   MC s.e. (bp)")
        for K in strikes:
            price, se = res[K]
            bs = BlackScholes(S=100.0, K=K, T=T)
            mc_iv = implied_vol(price, bs)
            se_iv = implied_vol(price + se, bs) - mc_iv
            hag = sabr_iv(100.0, K, T, p)
            print(f"  {K:6.1f}  {mc_iv:.6f}   {hag:.6f}    {1e4 * (hag - mc_iv):+10.2f}"
                  f"      {1e4 * se_iv:9.2f}")

    print("\nstep refinement at nu=0.40, T=1, K=100 (Hagan - MC, bp):")
    p = SABRParams(alpha=0.20, beta=1.0, rho=-0.30, nu=0.40)
    for n_steps in (125, 250, 500, 1000):
        res = sabr_conditional_mc(100.0, 1.0, p, (100.0,),
                                  n_pairs=60_000, n_steps=n_steps, seed=5)
        price, _ = res[100.0]
        mc_iv = implied_vol(price, BlackScholes(S=100.0, K=100.0, T=1.0))
        print(f"   {n_steps:5d} steps   {1e4 * (sabr_iv(100.0, 100.0, 1.0, p) - mc_iv):+7.2f}")


# ---------------------------------------------------------------------------
# 4. SVI against Gatheral-Jacquier and Lee
# ---------------------------------------------------------------------------
def svi_section() -> None:
    head("4. SVI against Gatheral-Jacquier and Lee")

    # (a) a flat slice must reproduce the Black-Scholes density exactly.
    sig, T = 0.25, 1.0
    flat = SVIRawParams(a=sig * sig * T, b=0.0, rho=0.0, m=0.0, sigma=0.1)

    def lognormal_in_k(k: float, w: float) -> float:
        d_minus = -k / math.sqrt(w) - 0.5 * math.sqrt(w)
        return math.exp(-0.5 * d_minus * d_minus) / math.sqrt(2.0 * math.pi * w)

    worst = max(abs(svi_density(k, flat) - lognormal_in_k(k, sig * sig * T))
                for k in np.linspace(-2.0, 2.0, 401))
    print(f"flat slice is lognormal:  max |svi_density - phi(d-)/sqrt(w)| = {worst:.3e}")

    # (b) the density is a unit mass and prices the forward, arbitrage or not.
    print("total mass and E[e^k] of the implied density:")
    for name, p in (("flat b=0", flat),
                    ("equity skew", SVIRawParams(a=0.012, b=0.10, rho=-0.60,
                                                 m=0.02, sigma=0.18)),
                    ("Vogt (min g < 0)", VOGT)):
        mass, mart = density_moments(p)
        print(f"   {name:18s} int p dk - 1 = {mass - 1.0:+.3e}   "
              f"int e^k p dk - 1 = {mart - 1.0:+.3e}")

    # (c) the wing limit of g, which is Lee's bound in raw-SVI coordinates.
    rng = np.random.default_rng(SEED)
    worst_wing = 0.0
    for _ in range(500):
        b = float(rng.uniform(0.02, 1.5))
        rho = float(rng.uniform(-0.95, 0.95))
        p = SVIRawParams(a=0.04, b=b, rho=rho, m=0.0, sigma=0.3)
        pred = 0.25 - (b * (1.0 + rho)) ** 2 / 16.0
        worst_wing = max(worst_wing, abs(svi_g(1e8, p) - pred))
    print(f"g(k) -> 1/4 - b^2(1+rho)^2/16:  max |g(1e8) - limit| over 500 draws "
          f"= {worst_wing:.3e}")

    flagged = 0
    trials = 300
    rng = np.random.default_rng(SEED + 1)
    for _ in range(trials):
        rho = float(rng.uniform(-0.90, 0.90))
        b = 2.0 / (1.0 + abs(rho)) * float(rng.uniform(1.05, 3.0))
        p = SVIRawParams(a=float(rng.uniform(0.01, 0.20)), b=b, rho=rho,
                         m=float(rng.uniform(-0.3, 0.3)),
                         sigma=float(rng.uniform(0.05, 0.8)))
        if svi_min_g(p, -60.0, 60.0, n=4001)[0] < 0.0:
            flagged += 1
    print(f"slices with b(1+|rho|) > 2 whose g dips negative: {flagged}/{trials}")

    # (d) the Gatheral-Jacquier sufficient conditions on SSVI.
    rng = np.random.default_rng(SEED + 2)
    drawn, violations, tightest = 0, 0, math.inf
    while drawn < 500:
        rho = float(rng.uniform(-0.98, 0.98))
        s = 1.0 + abs(rho)
        phi = float(rng.uniform(0.1, 8.0))
        theta = (4.0 / s) * float(rng.uniform(0.90, 0.999)) / phi
        if theta * phi * phi * s > 4.0:
            continue
        drawn += 1
        min_g = svi_min_g(ssvi_as_raw(theta, phi, rho), -40.0, 40.0, n=8001)[0]
        tightest = min(tightest, min_g)
        if min_g < 0.0:
            violations += 1
    print(f"SSVI slices inside the GJ conditions (drawn at 90-99.9% of the "
          f"first bound): {drawn}")
    print(f"   min g(k) over all of them = {tightest:+.6f}   violations = {violations}")

    # (e) the analytic and the discrete butterfly screens on the same slice.
    ks = [float(k) for k in np.linspace(0.40, 1.40, 21)]
    strikes = [100.0 * math.exp(k) for k in ks]
    ivs = [math.sqrt(svi_w(k, VOGT)) for k in ks]
    flagged_ks = [ks[i] for i in butterfly_violations(strikes, ivs, T=1.0,
                                                      forward=100.0, eps=1e-9)]
    negative = [k for k in np.linspace(0.40, 1.40, 2001) if svi_g(float(k), VOGT) < 0.0]
    print(f"Vogt slice: analytic g(k) < 0 on k in "
          f"[{min(negative):.3f}, {max(negative):.3f}]")
    print(f"            discrete screen flags k in "
          f"[{min(flagged_ks):.3f}, {max(flagged_ks):.3f}]  "
          f"({len(flagged_ks)} of {len(ks) - 2} interior strikes)")


# ---------------------------------------------------------------------------
# 5. the surface's calendar check against a brute-force scan
# ---------------------------------------------------------------------------
def surface_section() -> None:
    head("5. the fitted surface's calendar condition, brute-forced")
    truth = {
        1 / 12: SVIRawParams(a=-0.00087, b=0.030, rho=-0.75, m=0.0, sigma=0.10),
        0.25: SVIRawParams(a=0.00138, b=0.045, rho=-0.70, m=0.0, sigma=0.13),
        0.50: SVIRawParams(a=0.00692, b=0.058, rho=-0.65, m=0.0, sigma=0.16),
        1.00: SVIRawParams(a=0.02110, b=0.075, rho=-0.60, m=0.0, sigma=0.20),
        2.00: SVIRawParams(a=0.05400, b=0.100, rho=-0.55, m=0.0, sigma=0.26),
    }
    ks = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    surf = fit_svi_surface([(T, ks, [svi_w(k, p) for k in ks])
                            for T, p in truth.items()])
    grid_k = np.linspace(-0.6, 0.6, 241)
    grid_T = np.linspace(0.01, 2.5, 400)
    worst_drop = 0.0
    for k in grid_k:
        prev = surf.total_variance(float(k), float(grid_T[0]))
        for T in grid_T[1:]:
            cur = surf.total_variance(float(k), float(T))
            worst_drop = min(worst_drop, cur - prev)
            prev = cur
    print(f"brute force over {len(grid_k)}x{len(grid_T)} (k, T): "
          f"worst decrease in w = {worst_drop:.3e}")
    print(f"surf.calendar_arbitrage_free() = {surf.calendar_arbitrage_free()}")
    print(f"surf.butterfly_arbitrage_free() = {surf.butterfly_arbitrage_free()}")
    worst_iv = max(
        abs(surf.iv(k, T) - math.sqrt(svi_w(k, truth[T]) / T))
        for T in truth for k in ks
    )
    print(f"max |surface iv - the slice it was fitted to| = {worst_iv:.3e}")


def main() -> None:
    print("volsurf validation — every number below is printed by this script")
    black_scholes_section()
    sabr_limits_section()
    sabr_mc_section()
    svi_section()
    surface_section()
    print("\ndone")


if __name__ == "__main__":
    main()
