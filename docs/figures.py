"""Render every figure in the README from the library's own output.

Run:  python docs/figures.py        ->  docs/hero.png
                                        docs/surface.png
                                        docs/calibration.png

Nothing here is illustrative. Every curve is drawn from values `volsurf`
computes at run time, so a change in the library shows up in the pictures.
Needs the optional plotting extras (`pip install -e ".[plot]"`).
"""

from __future__ import annotations

import math
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")  # headless render

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volsurf import (  # noqa: E402
    SVIRawParams,
    fit_svi_slice,
    fit_svi_surface,
    svi_density,
    svi_g,
    svi_min_g,
    svi_w,
)
from volsurf.surface import _nelder_mead, _params_to_x, _x_to_params  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parent

INK = "#1a202c"
MUTED = "#7c8794"
GRID = "#e6eaef"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f7a55"
SAND = "#b7791f"

# Axel Vogt's slice (Gatheral & Jacquier 2014, sec. 3): the standard example
# of an SVI smile that looks fine and is not.
VOGT = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

# A steep single-name skew. b = 0.35 with sigma = 0.15 is a sharp elbow — the
# regime where a naive 5-D least-squares fit falls into a local minimum.
SKEW = SVIRawParams(a=0.01, b=0.35, rho=-0.70, m=0.05, sigma=0.15)

# An index-like arbitrage-free surface: ATM vol 16% at 1M rising to 20% at 2y,
# with the put skew flattening as maturity grows.
SURFACE = {
    1 / 12: SVIRawParams(a=-0.00087, b=0.030, rho=-0.75, m=0.0, sigma=0.10),
    0.25: SVIRawParams(a=0.00138, b=0.045, rho=-0.70, m=0.0, sigma=0.13),
    0.50: SVIRawParams(a=0.00692, b=0.058, rho=-0.65, m=0.0, sigma=0.16),
    1.00: SVIRawParams(a=0.02110, b=0.075, rho=-0.60, m=0.0, sigma=0.20),
    2.00: SVIRawParams(a=0.05400, b=0.100, rho=-0.55, m=0.0, sigma=0.26),
}
EXPIRY_LABEL = {1 / 12: "1M", 0.25: "3M", 0.5: "6M", 1.0: "1Y", 2.0: "2Y"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titlepad": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 100,
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _fit_quiet(ks, ws, **kw) -> SVIRawParams:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_svi_slice(ks, ws, **kw)


def _textbook_5d_fit(ks, ws) -> SVIRawParams:
    """A single-start 5-D Nelder-Mead fit — the obvious way to calibrate SVI.

    This is the baseline `fit_svi_slice` replaced: one fixed initial guess,
    all five parameters searched at once. It is not a straw man; it is what a
    direct reading of the SVI paper suggests, and it is what this library did
    before v0.3.
    """
    init = SVIRawParams(a=min(ws), b=0.1, rho=-0.3, m=0.0, sigma=0.1)

    def obj(x):
        p = _x_to_params(x)
        return sum((svi_w(k, p) - w) ** 2 for k, w in zip(ks, ws, strict=True))

    x, _ = _nelder_mead(obj, _params_to_x(init), max_iter=800)
    return _x_to_params(x)


# --------------------------------------------------------------------------
# hero — the butterfly condition, diagnosed and repaired
# --------------------------------------------------------------------------
def hero() -> None:
    ks_q = [-1.5 + 0.05 * i for i in range(61)]          # quoted log-moneyness
    ws_q = [svi_w(k, VOGT) for k in ks_q]
    fixed = _fit_quiet(ks_q, ws_q, butterfly_penalty=1e5)

    zoom_lo, zoom_hi = 0.50, 1.50
    ks = [-1.5 + 3.0 * i / 1200.0 for i in range(1201)]
    wing = [k for k in ks if zoom_lo <= k <= zoom_hi]

    iv_v = [100.0 * math.sqrt(svi_w(k, VOGT)) for k in ks]     # quoted at T = 1y
    iv_f = [100.0 * math.sqrt(svi_w(k, fixed)) for k in ks]
    gap = max(abs(a - b) for a, b in zip(iv_v, iv_f, strict=True))
    min_g, k_star = svi_min_g(VOGT, -1.5, 1.5, n=3001)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.4))
    for ax in (ax1, ax2, ax3):
        despine(ax)
        ax.set_xlabel("log-moneyness   k = log(K/F)")

    # -- the two smiles -----------------------------------------------------
    ax1.axvspan(zoom_lo, zoom_hi, color=SAND, alpha=0.10, lw=0)
    ax1.plot(ks, iv_v, color=RED, lw=2.0, label="quoted smile")
    ax1.plot(ks, iv_f, color=GREEN, lw=1.6, ls=(0, (5, 2)),
             label="nearest arbitrage-free smile")
    ax1.plot(ks_q[::2], [100.0 * math.sqrt(w) for w in ws_q[::2]], ls="none",
             marker="o", ms=3.2, mfc="white", mec=RED, mew=0.9)
    ax1.set_ylabel("implied vol at 1y   (%)")
    ax1.set_title("A quoted smile, and the nearest admissible one")
    ax1.legend(loc="upper left")
    ax1.annotate(f"{gap:.2f} vol points apart at the widest",
                 xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                 color=MUTED)
    ax1.annotate("the wing below", xy=((zoom_lo + zoom_hi) / 2, 0.95),
                 xycoords=("data", "axes fraction"), ha="center",
                 fontsize=8, color=SAND)

    # -- g(k) on the right wing ---------------------------------------------
    ax2.axhline(0.0, color=INK, lw=0.9)
    g_v = [svi_g(k, VOGT) for k in wing]
    g_f = [svi_g(k, fixed) for k in wing]
    ax2.plot(wing, g_v, color=RED, lw=2.0)
    ax2.plot(wing, g_f, color=GREEN, lw=1.6, ls=(0, (5, 2)))
    ax2.fill_between(wing, g_v, 0.0, where=[g < 0 for g in g_v],
                     color=RED, alpha=0.16, interpolate=True)
    ax2.plot([k_star], [min_g], marker="o", ms=5.0, color=RED, zorder=5)
    ax2.annotate(f"min g = {min_g:.4f}   at k = {k_star:.2f}",
                 xy=(k_star, min_g), xytext=(10, -15), textcoords="offset points",
                 fontsize=8.5, color=RED)
    ax2.set_xlim(zoom_lo, zoom_hi)
    ax2.set_ylim(-0.055, 0.135)
    ax2.set_ylabel("g(k)")
    ax2.set_title("Gatheral–Jacquier: admissible iff  g(k) ≥ 0")
    ax2.annotate("g peaks above 1.3 at the money;\nthe whole question is out here",
                 xy=(0.97, 0.92), xycoords="axes fraction", ha="right",
                 va="top", fontsize=8, color=MUTED)

    # -- the density it implies ---------------------------------------------
    dens_lo = 0.60
    tail = [k for k in ks if dens_lo <= k <= zoom_hi]
    p_v = [1e4 * svi_density(k, VOGT) for k in tail]
    p_f = [1e4 * svi_density(k, fixed) for k in tail]
    neg = [k for k, p in zip(tail, p_v, strict=True) if p < 0]
    i_low = min(range(len(tail)), key=lambda i: p_v[i])
    ax3.axhline(0.0, color=INK, lw=0.9)
    ax3.plot(tail, p_v, color=RED, lw=2.0)
    ax3.plot(tail, p_f, color=GREEN, lw=1.6, ls=(0, (5, 2)))
    ax3.fill_between(tail, p_v, 0.0, where=[p < 0 for p in p_v],
                     color=RED, alpha=0.24, interpolate=True)
    ax3.set_xlim(dens_lo, zoom_hi)
    ax3.set_ylabel("risk-neutral density   p(k)   ×10⁻⁴")
    ax3.set_title("… because otherwise it is not a probability")
    ax3.annotate(f"p(k) < 0 for {min(neg):.2f} < k < {max(neg):.2f}:\n"
                 f"that butterfly pays you to own it",
                 xy=(tail[i_low], p_v[i_low]), xytext=(0.34, 0.58),
                 textcoords="axes fraction", fontsize=8.5, color=RED,
                 arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.9})

    fig.tight_layout()
    _save(fig, "hero.png")
    print(f"  smiles differ by at most {gap:.4f} vol points")
    print(f"  min g(k):  quoted {min_g:+.6f}   repaired "
          f"{svi_min_g(fixed, -1.5, 1.5, n=3001)[0]:+.6f}")
    print(f"  min p(k):  quoted {min(p_v) * 1e-4:+.4e}   "
          f"repaired {min(p_f) * 1e-4:+.4e}")


# --------------------------------------------------------------------------
# surface — the calendar condition, drawn
# --------------------------------------------------------------------------
def surface() -> None:
    ks_q = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    slices = [(T, ks_q, [svi_w(k, p) for k in ks_q]) for T, p in SURFACE.items()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surf = fit_svi_surface(slices)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for ax in (ax1, ax2):
        despine(ax)

    # -- iso-vol map --------------------------------------------------------
    kk = np.linspace(-0.5, 0.5, 161)
    tt = np.exp(np.linspace(math.log(1 / 12), math.log(2.0), 161))
    iv = np.array([[surf.iv(float(k), float(T)) * 100.0 for k in kk] for T in tt])

    mesh = ax1.contourf(kk, tt, iv, levels=20, cmap="YlGnBu", alpha=0.62)
    lines = ax1.contour(kk, tt, iv, levels=[15, 20, 25, 30, 35, 40, 50],
                        colors=INK, linewidths=0.7, alpha=0.65)
    ax1.clabel(lines, fmt="%d%%", fontsize=7.5, colors=INK)
    for T in surf.expiries:
        ax1.axhline(T, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.45)
    ax1.set_yscale("log")
    ax1.set_yticks(list(surf.expiries))
    ax1.set_yticklabels([EXPIRY_LABEL[T] for T in surf.expiries])
    ax1.set_xlabel("log-moneyness   k = log(K/F)")
    ax1.set_ylabel("expiry (log scale)")
    ax1.set_title("The surface itself — dashes mark the five quoted expiries")
    ax1.grid(False)
    fig.colorbar(mesh, ax=ax1, pad=0.03, label="implied vol (%)")

    # -- calendar condition -------------------------------------------------
    ts = np.exp(np.linspace(math.log(0.02), math.log(2.0), 240))
    palette = ["#1a365d", "#2b6cb0", "#63a2d8", "#b7791f", "#c05621"]
    for k, colour in zip([-0.5, -0.25, 0.0, 0.25, 0.5], palette, strict=True):
        ax2.plot(ts, [surf.total_variance(k, float(T)) for T in ts],
                 color=colour, lw=1.8, label=f"k = {k:+.2f}")
    for T in surf.expiries:
        ax2.axvline(T, color=MUTED, lw=0.7, ls=(0, (2, 3)))
    ax2.axvspan(0.02, min(surf.expiries), color=SAND, alpha=0.09, lw=0)
    ax2.annotate("below the front expiry,\nw is scaled ∝ T",
                 xy=(0.021, 0.66), xycoords=("data", "axes fraction"),
                 fontsize=8, color=SAND)
    ax2.set_xscale("log")
    ax2.set_xticks([0.02, *surf.expiries])
    ax2.set_xticklabels(["0.02", *[f"{T:.2f}" for T in surf.expiries]])
    ax2.set_xlabel("expiry T (years, log scale)")
    ax2.set_ylabel("total implied variance   w(k, T)")
    ax2.set_title("Every strike's variance rises in T — no free calendar spread")
    ax2.legend(loc="upper left", ncols=2)

    fig.tight_layout()
    _save(fig, "surface.png")
    atm = ", ".join(f"{EXPIRY_LABEL[T]} {surf.iv(0.0, T) * 100:.1f}%"
                    for T in surf.expiries)
    print(f"  ATM term structure: {atm}")
    print(f"  calendar_arbitrage_free={surf.calendar_arbitrage_free()}   "
          f"butterfly_arbitrage_free={surf.butterfly_arbitrage_free(-0.5, 0.5)}")


# --------------------------------------------------------------------------
# calibration — why the quasi-explicit reduction is not a detail
# --------------------------------------------------------------------------
def calibration() -> None:
    ks_q = [-0.8 + 0.05 * i for i in range(33)]
    ws_q = [svi_w(k, SKEW) for k in ks_q]
    naive = _textbook_5d_fit(ks_q, ws_q)
    quasi = _fit_quiet(ks_q, ws_q)

    ks = [-1.2 + 3.4 * i / 1200.0 for i in range(1201)]
    w_true = [svi_w(k, SKEW) for k in ks]

    def rel(p):
        return [abs(svi_w(k, p) - w) / w for k, w in zip(ks, w_true, strict=True)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for ax in (ax1, ax2):
        despine(ax)
        ax.set_xlabel("log-moneyness   k = log(K/F)")

    ax1.axvspan(min(ks_q), max(ks_q), color=SAND, alpha=0.08, lw=0)
    ax1.plot(ks, w_true, color=INK, lw=3.0, alpha=0.20, label="true slice")
    ax1.plot(ks, [svi_w(k, naive) for k in ks], color=RED, lw=1.6,
             ls=(0, (5, 2)), label="single-start 5-D fit")
    ax1.plot(ks, [svi_w(k, quasi) for k in ks], color=BLUE, lw=1.6,
             label="quasi-explicit fit")
    ax1.plot(ks_q, ws_q, ls="none", marker="o", ms=3.2, mfc="white", mec=INK,
             mew=0.9, label="quoted nodes")
    ax1.set_ylabel("total implied variance   w(k)")
    ax1.set_title("Same 33 quotes, same objective, two answers")
    ax1.legend(loc="lower left")
    ax1.annotate("quoted range", xy=(0.0, 0.03), xycoords=("data", "axes fraction"),
                 ha="center", fontsize=8, color=SAND)

    axi = ax1.inset_axes((0.52, 0.46, 0.45, 0.47))
    zk = [k for k in ks if -0.15 <= k <= 0.35]
    axi.plot(zk, [svi_w(k, SKEW) for k in zk], color=INK, lw=3.0, alpha=0.20)
    axi.plot(zk, [svi_w(k, naive) for k in zk], color=RED, lw=1.4, ls=(0, (5, 2)))
    axi.plot(zk, [svi_w(k, quasi) for k in zk], color=BLUE, lw=1.4)
    axi.plot([k for k in ks_q if -0.15 <= k <= 0.35],
             [w for k, w in zip(ks_q, ws_q, strict=True) if -0.15 <= k <= 0.35],
             ls="none", marker="o", ms=2.8, mfc="white", mec=INK, mew=0.8)
    axi.set_title("the elbow: sigma collapses to 9e-4,\nso the fit is a corner",
                  fontsize=7.5, color=MUTED, pad=3)
    axi.tick_params(labelsize=7)
    despine(axi)

    ax2.axvspan(min(ks_q), max(ks_q), color=SAND, alpha=0.08, lw=0)
    ax2.semilogy(ks, rel(naive), color=RED, lw=1.6, ls=(0, (5, 2)),
                 label="single-start 5-D fit")
    ax2.semilogy(ks, rel(quasi), color=BLUE, lw=1.6, label="quasi-explicit fit")
    ax2.axhline(1e-2, color=MUTED, lw=0.8, ls=":")
    ax2.annotate("1% error", xy=(1.75, 1.4e-2), fontsize=8, color=MUTED)
    ax2.set_ylim(1e-12, 3.0)
    ax2.set_ylabel("relative error in w(k)")
    ax2.set_title("Eight orders of magnitude, decided by the initial guess")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    _save(fig, "calibration.png")
    err_n = max(abs(svi_w(k, naive) - w) / w for k, w in zip(ks_q, ws_q, strict=True))
    err_q = max(abs(svi_w(k, quasi) - w) / w for k, w in zip(ks_q, ws_q, strict=True))
    print(f"  single-start  sigma={naive.sigma:.3e}  rho={naive.rho:+.4f}  "
          f"max rel err on the quoted nodes {err_n:.2%}")
    print(f"  quasi-explicit sigma={quasi.sigma:.6f}  rho={quasi.rho:+.4f}  "
          f"max rel err on the quoted nodes {err_q:.2e}")


def _save(fig, name: str) -> None:
    out = DOCS / name
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(DOCS.parent)}")


def main() -> None:
    style()
    hero()
    surface()
    calibration()


if __name__ == "__main__":
    main()
