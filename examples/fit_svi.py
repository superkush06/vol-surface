"""Demonstrate SVI parameterisation and plot a slice."""

import argparse

from volsurf.svi import SVIRawParams, svi_iv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--out", default="svi_slice.png")
    args = ap.parse_args()

    p = SVIRawParams(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.3)
    ks = [-0.5 + 0.01 * i for i in range(101)]
    ivs = [svi_iv(k, args.T, p) for k in ks]

    print(f"params: {p}")
    print(f"{'k':>8} {'iv':>10}")
    for k, iv in list(zip(ks, ivs, strict=False))[::10]:
        print(f"{k:>8.2f} {iv:>10.5f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to render plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, ivs)
    ax.set_xlabel("log-moneyness k = log(K/F)")
    ax.set_ylabel("implied vol")
    ax.set_title(f"SVI smile @ T={args.T}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
