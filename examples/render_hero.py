"""Render the README hero image: a 3-D implied-volatility surface.

Run:  python examples/render_hero.py   ->  writes docs/demo.png

Fits a calendar-arbitrage-free SVI surface to several expiries, then renders
implied vol as a function of log-moneyness and time-to-expiry — the classic
"vol surface" picture, plus the per-expiry smiles beneath it.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volsurf.surface import fit_svi_surface  # noqa: E402
from volsurf.svi import SVIRawParams, svi_w  # noqa: E402


def main() -> None:
    # A realistic arb-free surface: total variance rises with T, negative skew.
    ks = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    truths = {
        0.10: SVIRawParams(a=0.010, b=0.10, rho=-0.35, m=0.0, sigma=0.18),
        0.25: SVIRawParams(a=0.022, b=0.11, rho=-0.32, m=0.0, sigma=0.20),
        0.50: SVIRawParams(a=0.040, b=0.12, rho=-0.30, m=0.0, sigma=0.22),
        1.00: SVIRawParams(a=0.075, b=0.13, rho=-0.28, m=0.0, sigma=0.24),
        2.00: SVIRawParams(a=0.140, b=0.14, rho=-0.25, m=0.0, sigma=0.26),
    }
    slices = [(T, ks, [svi_w(k, p) for k in ks]) for T, p in truths.items()]
    surf = fit_svi_surface(slices)

    expiries = np.array(surf.expiries)
    K = np.array(ks)
    KK, TT = np.meshgrid(K, expiries)
    IV = np.array([[surf.iv(k, T) for k in K] for T in expiries]) * 100.0

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(11, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(KK, TT, IV, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("expiry T (yrs)")
    ax.set_zlabel("implied vol %")
    ax.set_title("SVI implied-volatility surface")
    ax.view_init(elev=22, azim=-128)

    ax2 = fig.add_subplot(1, 2, 2)
    for T in expiries:
        ax2.plot(K, [surf.iv(k, T) * 100 for k in K], lw=1.4, label=f"T={T:g}y")
    ax2.set_xlabel("log-moneyness k")
    ax2.set_ylabel("implied vol %")
    ax2.set_title("per-expiry smiles (negative skew)")
    ax2.legend(fontsize=8)

    arb = "arbitrage-free" if surf.calendar_arbitrage_free() else "HAS ARBITRAGE"
    fig.suptitle(f"vol-surface — calibrated SVI surface ({arb})")
    fig.tight_layout()

    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  (calendar_arbitrage_free={surf.calendar_arbitrage_free()})")


if __name__ == "__main__":
    main()
