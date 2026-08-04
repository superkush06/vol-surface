"""Fit SABR to a slice, then walk the fitted vol through the money.

Run:  PYTHONPATH=. python3 examples/fit_sabr.py [--beta 0.5]

Two things worth seeing. First, the 3-parameter fit at fixed beta recovers
the generating parameters from nine strikes. Second — the part that is easy
to get wrong — the Hagan formula runs through a single code path, so walking
the strike towards the forward converges smoothly to the ATM vol instead of
stepping. Anything that differentiates the smile near the money (vega, skew,
a calibration gradient) depends on that.
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
    for K, mv in zip(strikes, market, strict=True):
        got = sabr_iv(F, K, T, fit)
        print(f"{K:>6.1f} {mv:>12.5f} {got:>10.5f} {got - mv:>+10.5f}")

    print()
    print("approaching the forward from above — no step, no special case")
    atm = sabr_iv(F, F, T, fit)
    print(f"{'K - F':>10} {'sabr_iv':>18} {'gap to ATM':>13}")
    for p in range(2, 14, 2):
        K = F * (1.0 + 10.0 ** -p)
        iv = sabr_iv(F, K, T, fit)
        print(f"{10.0 ** -p * F:>10.0e} {iv:>18.14f} {iv - atm:>13.2e}")


if __name__ == "__main__":
    main()
