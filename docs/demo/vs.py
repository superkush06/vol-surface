"""Driver the browser figures call into.

Everything here is `volsurf` doing the work. The page draws what these
functions hand back and derives nothing of its own, so a claim on the page is
a claim about the library.

The truth surface and the noise are lifted from `examples/fit_surface.py`
rather than reinvented, so the residuals the page reports are the same ones
the README quotes.
"""

from __future__ import annotations

import math
import random

from volsurf import (
    BlackScholes,
    IVSolverError,
    SABRParams,
    SVIRawParams,
    butterfly_violations,
    calendar_violations,
    fit_sabr,
    fit_svi_surface,
    implied_vol,
    sabr_iv,
    svi_density,
    svi_g,
    svi_iv,
    svi_min_g,
    svi_w,
)

TRUTH = {
    1 / 12: SVIRawParams(a=-0.00087, b=0.030, rho=-0.75, m=0.0, sigma=0.10),
    0.25: SVIRawParams(a=0.00138, b=0.045, rho=-0.70, m=0.0, sigma=0.13),
    0.50: SVIRawParams(a=0.00692, b=0.058, rho=-0.65, m=0.0, sigma=0.16),
    1.00: SVIRawParams(a=0.02110, b=0.075, rho=-0.60, m=0.0, sigma=0.20),
    2.00: SVIRawParams(a=0.05400, b=0.100, rho=-0.55, m=0.0, sigma=0.26),
}
LABEL = {1 / 12: "1M", 0.25: "3M", 0.5: "6M", 1.0: "1Y", 2.0: "2Y"}
NOISE_BPS = 25.0
KLO, KHI = -0.55, 0.55

# The screen deliberately looks wider than the plot. A slice can be clean
# everywhere a strike was quoted and still admit a butterfly out in the wing,
# which is exactly what Vogt's slice does: it passes on [-0.55, 0.55] and only
# fails once the scan reaches k = 0.88. Checking only where the quotes are
# would certify the repo's own counterexample as sound.
SLO, SHI = -1.5, 1.5


# ---------------------------------------------------------------------------
# 01: one slice, and the three things it implies
# ---------------------------------------------------------------------------

def slice_curves(a, b, rho, m, sigma, T=1.0, n=161):
    """Vol, g(k) and the risk-neutral density for one set of SVI parameters.

    These three are the same object seen three ways. The smile is what a desk
    quotes, g is the Gatheral-Jacquier condition that decides whether the
    quote is admissible, and the density is what the quote claims about where
    the stock can end up. Push the smile somewhere silly and the density goes
    negative, which is a probability below zero and therefore a portfolio that
    pays you to hold it.
    """
    p = SVIRawParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    ks = [KLO + (KHI - KLO) * i / (n - 1) for i in range(n)]
    iv, g, dens = [], [], []
    for k in ks:
        try:
            iv.append(round(svi_iv(k, T, p), 6))
        except Exception:
            iv.append(None)
        try:
            g.append(round(svi_g(k, p), 6))
        except Exception:
            g.append(None)
        try:
            dens.append(round(svi_density(k, p), 6))
        except Exception:
            dens.append(None)
    # A reader dragging the handles will leave the region where the slice is
    # even well defined, so nothing here may raise: total variance can go
    # non-positive, and then g and the density have nothing to say. Fall back
    # to the sampled grid and report the collapse rather than throwing.
    ok = [v for v in g if v is not None]
    try:
        gmin, kmin = svi_min_g(p, SLO, SHI, n=601)
    except ValueError:
        gmin = min(ok) if ok else float("nan")
        kmin = ks[g.index(gmin)] if ok else 0.0
    defined = len(ok) == len(g)
    bad = [i for i, v in enumerate(dens) if v is not None and v < 0]
    return {
        "ks": [round(k, 4) for k in ks],
        "iv": iv, "g": g, "density": dens,
        "min_g": round(gmin, 6), "min_g_k": round(kmin, 4),
        "arb_free": defined and gmin >= 0.0,
        "defined": defined,
        "bad_from": round(ks[bad[0]], 3) if bad else None,
        "bad_to": round(ks[bad[-1]], 3) if bad else None,
        "bad_frac": round(len(bad) / len(ks), 4),
        "atm_vol": round(svi_iv(0.0, T, p), 6) if iv[len(iv) // 2] else None,
    }


def truth_params(T=1.0):
    p = TRUTH[T]
    return {"a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma}


# ---------------------------------------------------------------------------
# 02: calibrate to quotes that are not clean
# ---------------------------------------------------------------------------

def shaped(p, level=1.0, skew=1.0, wings=1.0):
    """The truth slice with its level, skew and wings scaled.

    `level` moves total variance up and down, `skew` leans rho toward or away
    from the downside, and `wings` steepens b. Together they are enough to
    walk the whole surface from a flat sheet to a pronounced smirk without
    leaving the region where SVI is well defined.
    """
    return SVIRawParams(
        a=p.a * level,
        b=max(0.004, p.b * wings),
        rho=max(-0.985, min(0.985, p.rho * skew)),
        m=p.m,
        sigma=p.sigma,
    )


def quoted_slices(noise_bps=NOISE_BPS, seed=20260727,
                  level=1.0, skew=1.0, wings=1.0):
    rng = random.Random(seed)
    ks = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    out = []
    for T, p0 in TRUTH.items():
        p = shaped(p0, level, skew, wings)
        vols = [math.sqrt(max(1e-8, svi_w(k, p)) / T)
                + rng.uniform(-1.0, 1.0) * noise_bps * 1e-4 for k in ks]
        out.append((T, ks, [v * v * T for v in vols], vols))
    return out


def fit_surface(noise_bps=NOISE_BPS, seed=20260727):
    """Fit all five expiries and report how far off each one lands, in bp."""
    sl = quoted_slices(noise_bps, seed)
    surf = fit_svi_surface([(T, ks, ws) for T, ks, ws, _ in sl])
    rows = []
    for (T, ks, _, vols), p in zip(sl, surf.params, strict=True):
        rms = math.sqrt(sum((surf.iv(k, T) - v) ** 2
                            for k, v in zip(ks, vols, strict=True)) / len(ks))
        rows.append({
            "T": round(T, 4), "label": LABEL[T],
            "a": round(p.a, 5), "b": round(p.b, 4), "rho": round(p.rho, 4),
            "m": round(p.m, 4), "sigma": round(p.sigma, 4),
            "atm": round(surf.iv(0.0, T), 5),
            "rms_bp": round(rms * 1e4, 1),
            "quotes": [round(v, 6) for v in vols],
            "fitted": [round(surf.iv(k, T), 6) for k in ks],
            "ks": ks,
        })
    return {
        "rows": rows,
        "noise_bps": noise_bps,
        "calendar_free": _try(surf.calendar_arbitrage_free),
        "butterfly_free": _try(lambda: surf.butterfly_arbitrage_free(SLO, SHI, n=601)),
        "worst_bp": max(r["rms_bp"] for r in rows),
        "best_bp": min(r["rms_bp"] for r in rows),
    }


def surface_grid(nk=49, nt=41, level=1.0, skew=1.0, wings=1.0):
    """Implied vol over (log-moneyness, expiry), for the opening figure."""
    # A shape the reader chose may be one the fitter cannot land, and this
    # must degrade rather than raise: the page reports the failure and keeps
    # the last good surface on screen.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sl = quoted_slices(level=level, skew=skew, wings=wings)
            surf = fit_svi_surface([(T, ks, ws) for T, ks, ws, _ in sl])
        except Exception as exc:
            return {"ok": False, "why": str(exc)[:120]}
    ks = [KLO + (KHI - KLO) * i / (nk - 1) for i in range(nk)]
    lo, hi = 1 / 12, 2.0
    ts = [lo * (hi / lo) ** (j / (nt - 1)) for j in range(nt)]
    grid = []
    for T in ts:
        row = []
        for k in ks:
            try:
                row.append(round(surf.iv(k, T), 6))
            except Exception:
                row.append(None)
        grid.append(row)
    flat = [v for row in grid for v in row if v is not None]
    if not flat:
        return {"ok": False, "why": "no admissible slice at this shape"}
    fill = sum(v is not None for v in
               (x for row in grid for x in row)) / (len(ts) * len(ks))
    return {
        "ok": True,
        "fill": round(fill, 4),
        "ks": [round(k, 4) for k in ks],
        "ts": [round(t, 5) for t in ts],
        "grid": grid,
        "vmin": round(min(flat), 6), "vmax": round(max(flat), 6),
        "quoted_T": [round(t, 5) for t in TRUTH],
        "labels": [LABEL[t] for t in TRUTH],
        "calendar_free": _try(surf.calendar_arbitrage_free),
        "butterfly_free": _try(lambda: surf.butterfly_arbitrage_free(SLO, SHI, n=601)),
        "atm_1y": _try(lambda: round(surf.iv(0.0, 1.0), 5)),
        "skew_1y": _try(lambda: round(surf.iv(-0.2, 1.0) - surf.iv(0.2, 1.0), 5)),
    }


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 03: the other model on the same quotes
# ---------------------------------------------------------------------------

def sabr_fit(T=1.0, F=100.0, beta=0.5, noise_bps=NOISE_BPS, seed=20260727):
    """Fit SABR to the same slice SVI just fitted, and score both in bp."""
    sl = {row[0]: row for row in quoted_slices(noise_bps, seed)}
    _, ks, ws, vols = sl[T]
    strikes = [F * math.exp(k) for k in ks]
    p = fit_sabr(F, T, strikes, vols, beta=beta)
    fitted = [sabr_iv(F, K, T, p) for K in strikes]
    rms = math.sqrt(sum((f - v) ** 2 for f, v in zip(fitted, vols, strict=True))
                    / len(vols))
    surf = fit_svi_surface([(T, ks, ws)])
    svi_line = [surf.iv(k, T) for k in ks]
    svi_rms = math.sqrt(sum((f - v) ** 2
                            for f, v in zip(svi_line, vols, strict=True)) / len(vols))
    return {
        "ks": ks, "quotes": [round(v, 6) for v in vols],
        "sabr": [round(v, 6) for v in fitted],
        "svi": [round(v, 6) for v in svi_line],
        "alpha": round(p.alpha, 5), "beta": round(p.beta, 3),
        "rho": round(p.rho, 4), "nu": round(p.nu, 4),
        "sabr_bp": round(rms * 1e4, 1), "svi_bp": round(svi_rms * 1e4, 1),
        "T": T, "F": F,
    }


def sabr_curve(alpha, beta, rho, nu, T=1.0, F=100.0,
               noise_bps=NOISE_BPS, seed=20260727):
    """A SABR smile at parameters the reader chose, scored against the quotes.

    Same 21 marks the fitter saw, so the error here is comparable with the
    fitted number rather than being a different measurement.
    """
    p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    sl = {row[0]: row for row in quoted_slices(noise_bps, seed)}
    _, ks, _, vols = sl[T]
    at_quotes, bad = [], False
    for k in ks:
        try:
            at_quotes.append(sabr_iv(F, F * math.exp(k), T, p))
        except Exception:
            at_quotes.append(None); bad = True
    good = [(f, v) for f, v in zip(at_quotes, vols, strict=True) if f is not None]
    rms = (math.sqrt(sum((f - v) ** 2 for f, v in good) / len(good))
           if good else float("nan"))
    fine = [KLO + (KHI - KLO) * i / 120 for i in range(121)]
    curve = []
    for k in fine:
        try:
            curve.append(round(sabr_iv(F, F * math.exp(k), T, p), 6))
        except Exception:
            curve.append(None)
    return {
        "ks": [round(k, 4) for k in fine], "iv": curve,
        "at_quotes": [None if v is None else round(v, 6) for v in at_quotes],
        "bp": round(rms * 1e4, 1) if good else None,
        "defined": not bad,
        "atm": round(sabr_iv(F, F, T, p), 6) if not bad else None,
    }


# ---------------------------------------------------------------------------
# 00a: the inversion everything else stands on
# ---------------------------------------------------------------------------

def price_and_invert(S=100.0, K=100.0, T=1.0, r=0.02, q=0.0, sigma=0.20,
                     call=True):
    """Price an option, then recover the volatility from that price.

    This is the foundation the rest of the library sits on. Every quoted vol
    on the surface is the output of exactly this inversion, run once per mark.
    `BlackScholes` prices in closed form and exposes delta, vega and gamma;
    then
    `implied_vol` throws the price back at a Brent solver and asks what sigma
    produced it. Recovering the input to solver tolerance is the check that
    the two halves agree.
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, q=q)
    px = bs.price(sigma, call=call)
    try:
        back = implied_vol(px, bs, call=call)
        err = abs(back - sigma)
        failed = None
    except IVSolverError as exc:
        back, err, failed = None, None, str(exc)[:160]
    g = {"delta": bs.delta(sigma, call=call),
         "vega": bs.vega(sigma),
         "gamma": bs.gamma(sigma)}
    return {
        "price": round(px, 6),
        "sigma_in": round(sigma, 6),
        "sigma_out": None if back is None else round(back, 10),
        "abs_err": None if err is None else err,
        "failed": failed,
        "call": call,
        "intrinsic": round(max(0.0, (S - K) if call else (K - S)), 4),
        "greeks": {k: round(v, 6) for k, v in g.items()},
    }


def price_curve(S=100.0, T=1.0, r=0.02, q=0.0, sigma=0.20, call=True, n=61):
    """Price against strike, and the vol recovered back out at each one.

    Drawn together so the round trip is visible rather than asserted: the
    price falls away across strikes while the recovered vol stays flat at the
    number that generated it.
    """
    ks = [-0.5 + 1.0 * i / (n - 1) for i in range(n)]
    px, back = [], []
    for k in ks:
        bs = BlackScholes(S=S, K=S * math.exp(k), T=T, r=r, q=q)
        p = bs.price(sigma, call=call)
        px.append(round(p, 6))
        try:
            back.append(round(implied_vol(p, bs, call=call), 8))
        except IVSolverError:
            back.append(None)
    good = [v for v in back if v is not None]
    return {
        "ks": [round(k, 4) for k in ks], "price": px, "iv": back,
        "sigma": sigma,
        "worst_err": max(abs(v - sigma) for v in good) if good else None,
        "recovered": len(good), "n": n,
    }


def screen_quotes(noise_bps=NOISE_BPS, seed=20260727, T=1.0, F=100.0):
    """Run the discrete no-arbitrage screens over the raw quotes.

    `butterfly_violations` and `calendar_violations` work on strikes and vols
    directly, with no model in between. That makes them the check you can run
    on a screen before deciding whether it is even worth fitting, which is a
    different job from the parametric g(k) test the fitted slice gets.
    """
    sl = {row[0]: row for row in quoted_slices(noise_bps, seed)}
    _, ks, _, vols = sl[T]
    strikes = [F * math.exp(k) for k in ks]
    bfly = butterfly_violations(strikes, vols, T, F)
    short_T = 0.5
    _, ks_s, _, vols_s = sl[short_T]
    cal = calendar_violations(strikes, vols_s, vols, short_T, T, F)
    return {
        "ks": ks, "strikes": [round(s, 3) for s in strikes],
        "vols": [round(v, 6) for v in vols],
        "butterfly_idx": bfly, "calendar_idx": cal,
        "n": len(ks), "noise_bps": noise_bps,
        "short_label": LABEL[short_T], "long_label": LABEL[T],
    }
