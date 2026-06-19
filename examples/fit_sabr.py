"""Fit SABR (equity-style, beta=1.0) to a synthetic market slice.

Run:  PYTHONPATH=. python3 examples/fit_sabr.py
"""

import argparse

from volsurf.sabr import SABRParams, sabr_iv
from volsurf.sabr_fit import fit_sabr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0,
                    help="0.5 for rates/FX, 1.0 (default) for equity/index")
    args = ap.parse_args()

    truth = SABRParams(alpha=0.25, beta=args.beta, rho=-0.35, nu=0.55)
    F, T = 100.0, 1.0
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]
    market = [sabr_iv(F, K, T, truth) for K in strikes]

    fit = fit_sabr(F, T, strikes, market, beta=args.beta, max_iter=500)
    print(f"truth: alpha={truth.alpha:.4f}, beta={truth.beta:.2f}, "
          f"rho={truth.rho:+.4f}, nu={truth.nu:.4f}")
    print(f"fit:   alpha={fit.alpha:.4f}, beta={fit.beta:.2f}, "
          f"rho={fit.rho:+.4f}, nu={fit.nu:.4f}")
    print()
    print(f"{'K':>6} {'market_iv':>12} {'fit_iv':>10} {'err':>10}")
    for K, mv in zip(strikes, market, strict=False):
        got = sabr_iv(F, K, T, fit)
        print(f"{K:>6.1f} {mv:>12.5f} {got:>10.5f} {got - mv:>+10.5f}")


if __name__ == "__main__":
    main()
