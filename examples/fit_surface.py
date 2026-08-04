"""Calibrate a five-expiry surface from quoted vols, then use it.

Run:  PYTHONPATH=. python3 examples/fit_surface.py

The story: a market maker publishes 21 strikes at each of five expiries. We
convert those quotes to total variance, fit one raw-SVI slice per expiry with
the quasi-explicit method, check both no-arbitrage conditions, and then ask
the surface for a vol at an expiry nobody quoted.

The quotes are generated from a known surface and perturbed by a fixed
pseudo-random seed, so the residuals below are what a real fit to slightly
noisy marks looks like — not a synthetic exact recovery.
"""

from __future__ import annotations

import math
import random

from volsurf import SVIRawParams, fit_svi_surface, svi_w

TRUTH = {
    1 / 12: SVIRawParams(a=-0.00087, b=0.030, rho=-0.75, m=0.0, sigma=0.10),
    0.25: SVIRawParams(a=0.00138, b=0.045, rho=-0.70, m=0.0, sigma=0.13),
    0.50: SVIRawParams(a=0.00692, b=0.058, rho=-0.65, m=0.0, sigma=0.16),
    1.00: SVIRawParams(a=0.02110, b=0.075, rho=-0.60, m=0.0, sigma=0.20),
    2.00: SVIRawParams(a=0.05400, b=0.100, rho=-0.55, m=0.0, sigma=0.26),
}
LABEL = {1 / 12: "1M", 0.25: "3M", 0.5: "6M", 1.0: "1Y", 2.0: "2Y"}
NOISE_BPS = 25.0  # +/- 25 bp of vol on every quote


def quoted_slices():
    """(T, ks, total_vars) with a quarter-vol-point of noise on each mark."""
    rng = random.Random(20260727)
    ks = [round(-0.5 + 0.05 * i, 3) for i in range(21)]
    out = []
    for T, p in TRUTH.items():
        vols = []
        for k in ks:
            iv = math.sqrt(svi_w(k, p) / T)
            vols.append(iv + rng.uniform(-1.0, 1.0) * NOISE_BPS * 1e-4)
        out.append((T, ks, [v * v * T for v in vols], vols))
    return out


def main() -> None:
    slices = quoted_slices()
    surf = fit_svi_surface([(T, ks, ws) for T, ks, ws, _ in slices])

    print("fitted slices")
    print(f"{'expiry':>7} {'a':>10} {'b':>8} {'rho':>8} {'m':>8} {'sigma':>8}"
          f" {'ATM vol':>9} {'rms err':>9}")
    for (T, ks, _, vols), p in zip(slices, surf.params, strict=True):
        rms = math.sqrt(sum((surf.iv(k, T) - v) ** 2
                            for k, v in zip(ks, vols, strict=True)) / len(ks))
        print(f"{LABEL[T]:>7} {p.a:>10.5f} {p.b:>8.4f} {p.rho:>+8.4f}"
              f" {p.m:>+8.4f} {p.sigma:>8.4f} {surf.iv(0.0, T):>8.2%}"
              f" {rms * 1e4:>7.1f}bp")

    print()
    print(f"calendar arbitrage free:  {surf.calendar_arbitrage_free()}")
    print(f"butterfly arbitrage free: {surf.butterfly_arbitrage_free(-0.5, 0.5)}")

    print()
    print("interpolated where nobody quotes (9M, between the 6M and 1Y slices)")
    print(f"{'k':>7} {'K/F':>7} {'implied vol':>13}")
    for k in (-0.4, -0.2, 0.0, 0.2, 0.4):
        print(f"{k:>+7.2f} {math.exp(k):>7.3f} {surf.iv(k, 0.75):>12.2%}")


if __name__ == "__main__":
    main()
