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
    SABRParams,
    SVIRawParams,
    fit_sabr,
    fit_svi_surface,
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
        gmin, kmin = svi_min_g(p, KLO, KHI)
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

def quoted_slices(noise_bps=NOISE_BPS, seed=20260727):
    rng = random.Random(seed)
    ks = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    out = []
    for T, p in TRUTH.items():
        vols = [math.sqrt(svi_w(k, p) / T)
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
        "calendar_free": surf.calendar_arbitrage_free(),
        "butterfly_free": surf.butterfly_arbitrage_free(KLO, KHI),
        "worst_bp": max(r["rms_bp"] for r in rows),
        "best_bp": min(r["rms_bp"] for r in rows),
    }


def surface_grid(nk=49, nt=41):
    """Implied vol over (log-moneyness, expiry), for the opening figure."""
    sl = quoted_slices()
    surf = fit_svi_surface([(T, ks, ws) for T, ks, ws, _ in sl])
    ks = [KLO + (KHI - KLO) * i / (nk - 1) for i in range(nk)]
    lo, hi = 1 / 12, 2.0
    ts = [lo * (hi / lo) ** (j / (nt - 1)) for j in range(nt)]
    grid = [[round(surf.iv(k, T), 6) for k in ks] for T in ts]
    flat = [v for row in grid for v in row]
    return {
        "ks": [round(k, 4) for k in ks],
        "ts": [round(t, 5) for t in ts],
        "grid": grid,
        "vmin": round(min(flat), 6), "vmax": round(max(flat), 6),
        "quoted_T": [round(t, 5) for t in TRUTH],
        "labels": [LABEL[t] for t in TRUTH],
    }


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


def sabr_curve(alpha, rho, nu, beta=0.5, T=1.0, F=100.0, n=121):
    """SABR smile for arbitrary parameters, so the page can expose the knobs."""
    p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    ks = [KLO + (KHI - KLO) * i / (n - 1) for i in range(n)]
    out = []
    for k in ks:
        try:
            out.append(round(sabr_iv(F, F * math.exp(k), T, p), 6))
        except Exception:
            out.append(None)
    return {"ks": [round(k, 4) for k in ks], "iv": out,
            "atm": out[n // 2] if out[n // 2] is not None else None}
