"""Render the demo's link-preview card.

The image a scanner shows when the URL is pasted into Slack, LinkedIn or a
message is the only thing most people will ever see of the page, so it should
be the page. This fits the same surface the demo fits and draws it with the
same orthographic projection the browser uses, so the card and the page are
the same picture.

    python docs/demo/make_card.py        # writes docs/demo/card.png

Needs the `plot` extra (matplotlib). Open Graph wants 1200x630.
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vs  # noqa: E402

W, H = 1200, 630
PAPER, INK, QUIET, HAIR = "#F7F4EF", "#35322C", "#8B857A", "#DAD4C9"
SERIF = ["Newsreader", "Georgia", "Iowan Old Style", "DejaVu Serif"]
MONO = ["IBM Plex Mono", "Menlo", "DejaVu Sans Mono"]

# the page's volTint ramp, as fractions
RAMP = [(0.953, 0.937, 0.906), (0.839, 0.780, 0.667), (0.757, 0.580, 0.376),
        (0.588, 0.353, 0.227), (0.325, 0.188, 0.157)]
YAW, PITCH = -0.66, 0.58
ZMAX = 0.80


def tint(t: float) -> tuple[float, float, float]:
    x = max(0.0, min(0.999, t)) * (len(RAMP) - 1)
    i = int(x)
    f = x - i
    a, b = RAMP[i], RAMP[i + 1]
    return tuple(a[c] + (b[c] - a[c]) * f for c in range(3))


def raw(x: float, y: float, z: float) -> tuple[float, float, float]:
    cy, sy = math.cos(YAW), math.sin(YAW)
    cp, sp = math.cos(PITCH), math.sin(PITCH)
    rx = x * cy - y * sy
    ry = x * sy + y * cy
    return rx, ry * sp - z * cp, ry * cp + z * sp


def main() -> None:
    g = vs.surface_grid(nk=45, nt=35)
    if not g["ok"]:
        raise SystemExit("surface did not fit")
    ks, ts, grid = g["ks"], g["ts"], g["grid"]
    vmin, vmax = g["vmin"], g["vmax"]
    nk, nt = len(ks), len(ts)
    lt0, lt1 = math.log(ts[0]), math.log(ts[-1])

    def NX(i): return -1 + 2 * i / (nk - 1)
    def NY(j): return -1 + 2 * (math.log(ts[j]) - lt0) / (lt1 - lt0)
    def NZ(v): return ZMAX * (v - vmin) / (vmax - vmin)

    # fit the projected corners into the panel the surface gets
    px0, px1, py0, py1 = 560, 1176, 92, 566
    us, vsy = [], []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (0, ZMAX):
                u, vv, _ = raw(x, y, z)
                us.append(u); vsy.append(vv)
    s = min((px1 - px0) / (max(us) - min(us)), (py1 - py0) / (max(vsy) - min(vsy)))
    ox = (px0 + px1) / 2 - (min(us) + max(us)) / 2 * s
    oy = (py0 + py1) / 2 - (min(vsy) + max(vsy)) / 2 * s

    def proj(x, y, z):
        u, vv, d = raw(x, y, z)
        return (ox + u * s, oy + vv * s, d)

    quads, colours, depths = [], [], []
    for j in range(nt - 1):
        for i in range(nk - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if any(grid[b][a] is None for a, b in corners):
                continue
            pts = [proj(NX(a), NY(b), NZ(grid[b][a])) for a, b in corners]
            mean = sum(grid[b][a] for a, b in corners) / 4
            quads.append([(p[0], p[1]) for p in pts])
            colours.append(tint((mean - vmin) / (vmax - vmin)))
            depths.append((pts[0][2] + pts[2][2]) / 2)

    order = sorted(range(len(quads)), key=lambda i: depths[i])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    ax.add_collection(PolyCollection(
        [quads[i] for i in order], facecolors=[colours[i] for i in order],
        edgecolors=(0.21, 0.20, 0.17, 0.10), linewidths=0.4))

    # the ridges that were actually quoted
    for T, lab in zip(g["quoted_T"], g["labels"], strict=True):
        j = min(range(nt), key=lambda jj: abs(math.log(ts[jj]) - math.log(T)))
        line = [proj(NX(i), NY(j), NZ(grid[j][i]) + 0.004)
                for i in range(nk) if grid[j][i] is not None]
        if len(line) < 2:
            continue
        ax.plot([p[0] for p in line], [p[1] for p in line], color="#4A453C", lw=1.4)
        ax.text(line[-1][0] + 8, line[-1][1], lab, family=MONO, fontsize=10,
                color=INK, va="center")

    ax.text(64, 92, "A N   I N T E R A C T I V E   V I S U A L I S A T I O N",
            family=MONO, fontsize=12, color=QUIET, va="center")
    ax.text(62, 176, "vol-surface", family=SERIF, fontsize=62, color=INK, va="center")
    # three short lines, so none of them reaches the surface's leading edge
    for y, line in ((228, "an arbitrage-free"), (274, "volatility surface"),
                    (320, "you can break")):
        ax.text(62, y, line, family=SERIF, fontsize=38, color=INK, va="center")
    for y, line in ((376, "Drag the five SVI parameters until the slice"),
                    (404, "implies negative risk-neutral density, and"),
                    (432, "watch the butterfly check catch it.")):
        ax.text(64, y, line, family=SERIF, fontsize=19, color="#5E594F", va="center")

    fit = vs.fit_surface()
    ax.plot([64, 500], [468, 468], color=HAIR, lw=1)
    for i, (k, v) in enumerate([
            ("EXPIRIES", "5"), ("STRIKES EACH", "21"),
            ("RMS ERROR", f"{fit['best_bp']:.1f}–{fit['worst_bp']:.1f}bp")]):
        x = 64 + i * 150
        ax.text(x, 496, k, family=MONO, fontsize=9.5, color=QUIET, va="center")
        ax.text(x, 522, v, family=MONO, fontsize=17, color=INK, va="center")

    ax.plot([64, W - 64], [H - 62, H - 62], color=HAIR, lw=1)
    ax.text(64, H - 36, "superkush06.github.io/vol-surface/demo", family=MONO,
            fontsize=13, color=QUIET, va="center")
    ax.text(W - 64, H - 36, "SVI + SABR CALIBRATION IN THE BROWSER", family=MONO,
            fontsize=13, color=QUIET, va="center", ha="right")

    out = pathlib.Path(__file__).resolve().parent / "card.png"
    fig.savefig(out, facecolor=PAPER)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
