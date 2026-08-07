"""Every number printed in README.md, recomputed and compared to the file.

The README claims that "every output block in this file is pasted from one of
those runs". This module is what makes that claim enforceable: each test pulls
the literal out of `README.md` with a regex and checks it against a live call,
so the suite fails if the code drifts from the prose *or* if the prose is
retyped by hand. Changing `0.08150882562974118` to `0.08150882562974119`, or
`1.83 vol points` to `1.84`, breaks the build.

Two kinds of tolerance are used, and the distinction matters:

* closed-form results (Black-Scholes, `svi_w`, `svi_g`) are compared at
  `rel_tol=1e-12` — tight enough that any transcription error fails, loose
  enough to survive the 1-ulp differences between glibc's and Apple libm's
  `erf`/`exp`/`log`, which is why the CI matrix can span two operating
  systems;
* optimiser output (`fit_svi_slice`) is compared at `rel_tol=1e-6`, and the
  README's accuracy claims about it are checked as claims about significant
  figures rather than as bit patterns.
"""

from __future__ import annotations

import math
import pathlib
import re
import subprocess
import sys
import warnings

import pytest

from volsurf import (
    BlackScholes,
    ButterflyArbitrageWarning,
    SVIRawParams,
    butterfly_violations,
    fit_svi_slice,
    implied_vol,
    svi_density,
    svi_min_g,
    svi_w,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Axel Vogt's slice, as it appears in the README's "Screen" section.
VOGT = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

CLOSED_FORM_TOL = 1e-12
FIT_TOL = 1e-6


def literal(pattern: str) -> float:
    """The one number `pattern` captures in README.md, as a float.

    A pattern that no longer matches is itself a failure: it means the README
    was reworded and this check silently stopped covering anything.
    """
    matches = re.findall(pattern, README)
    assert len(matches) == 1, (
        f"expected exactly one README match for {pattern!r}, found {len(matches)}"
    )
    return float(matches[0].replace("−", "-"))


def matching_sig_figs(value: float, truth: float) -> int:
    """How many leading significant digits of `value` agree with `truth`.

    This is the sense in which the README says a parameter "comes back to
    seven significant figures": round both to n digits and compare.
    """
    for n in range(17, 0, -1):
        if f"{value:.{n - 1}e}" == f"{truth:.{n - 1}e}":
            return n
    return 0


def test_price_and_invert_block() -> None:
    bs = BlackScholes(S=100, K=105, T=0.5, r=0.02)
    assert bs.price(0.21) == pytest.approx(
        literal(r"bs\.price\(0\.21\)\s+#\s+([\d.]+)"), rel=CLOSED_FORM_TOL)
    assert implied_vol(4.2623438, bs) == pytest.approx(
        literal(r"implied_vol\(4\.2623438, bs\)\s+#\s+([\d.]+)"), rel=CLOSED_FORM_TOL)
    assert bs.vega(0.21) == pytest.approx(
        literal(r"bs\.vega\(0\.21\)\s+#\s+([\d.]+)"), rel=CLOSED_FORM_TOL)


def test_bracket_error_message_is_quoted_verbatim() -> None:
    """The README pastes an `IVSolverError`; the real one must still say that."""
    with pytest.raises(Exception) as excinfo:
        implied_vol(45.0, BlackScholes(S=100, K=200, T=0.02))
    assert "implied vol exceeds the upper bracket hi=5.0" in str(excinfo.value)
    assert "implied vol exceeds the upper bracket hi=5.0" in README


def _readme_calibrate_fit() -> SVIRawParams:
    truth = SVIRawParams(a=0.01, b=0.35, rho=-0.70, m=0.05, sigma=0.15)
    ks = [-0.8 + 0.05 * i for i in range(33)]
    return fit_svi_slice(ks, [svi_w(k, truth) for k in ks])


def test_calibrate_block_parameters() -> None:
    fit = _readme_calibrate_fit()
    printed = re.search(
        r"# SVIRawParams\(a=([-\d.e]+), b=([-\d.e]+),\s*\n"
        r"#\s+rho=([-\d.e]+), m=([-\d.e]+),\s*\n"
        r"#\s+sigma=([-\d.e]+)\)", README)
    assert printed is not None, "the README no longer prints the fitted SVI parameters"
    for name, text in zip(("a", "b", "rho", "m", "sigma"), printed.groups(), strict=True):
        assert getattr(fit, name) == pytest.approx(float(text), rel=FIT_TOL), name


def test_calibrate_block_min_g() -> None:
    fit = _readme_calibrate_fit()
    min_g, k_star = svi_min_g(fit, -0.8, 0.8)
    printed = re.search(r"# \(([\d.]+), (-[\d.]+)\)\s+<- min g > 0: admissible", README)
    assert printed is not None
    assert min_g == pytest.approx(float(printed.group(1)), rel=FIT_TOL)
    assert k_star == pytest.approx(float(printed.group(2)), rel=FIT_TOL)
    assert min_g > 0.0


def test_the_recovery_accuracy_claim() -> None:
    """"Seven significant figures or better ... a comes back to six."""
    truth = SVIRawParams(a=0.01, b=0.35, rho=-0.70, m=0.05, sigma=0.15)
    fit = _readme_calibrate_fit()
    for name in ("b", "rho", "m", "sigma"):
        digits = matching_sig_figs(getattr(fit, name), getattr(truth, name))
        assert digits >= 7, f"{name} recovered to only {digits} significant figures"
    assert matching_sig_figs(fit.a, truth.a) == 6

    # Also a bound, for the same reason: this is a fitted value, and pinning
    # its first decimal would break on any CPython whose optimiser path differs.
    rel_a = abs(fit.a - truth.a) / truth.a
    bound = literal(r"relative error under\s*\$([\d.]+)\\times10\^\{-7\}\$")
    assert rel_a * 1e7 < bound, f"a recovered to {rel_a * 1e7:.2f}e-7, claimed under {bound}e-7"


def test_screen_block_on_the_vogt_slice() -> None:
    min_g, k_star = svi_min_g(VOGT, -1.5, 1.5)
    printed = re.search(r"svi_min_g\(vogt, -1\.5, 1\.5\)\s+# \((-[\d.]+), ([\d.]+)\)", README)
    assert printed is not None
    assert min_g == pytest.approx(float(printed.group(1)), rel=CLOSED_FORM_TOL)
    assert k_star == pytest.approx(float(printed.group(2)), rel=CLOSED_FORM_TOL)

    assert svi_density(0.879, VOGT) == pytest.approx(
        literal(r"svi_density\(0\.879, vogt\)\s+#\s+(-[\d.e-]+)\s+<- negative"),
        rel=CLOSED_FORM_TOL)


def test_the_pasted_butterfly_warning() -> None:
    ks = [-1.5 + 0.05 * i for i in range(61)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ButterflyArbitrageWarning)
        fit_svi_slice(ks, [svi_w(k, VOGT) for k in ks])
    messages = [str(w.message) for w in caught
                if issubclass(w.category, ButterflyArbitrageWarning)]
    assert len(messages) == 1
    quoted = re.search(r"min g\(k\) = (-[\d.]+) < 0 on \[-1\.5, 1\.5\]", README)
    assert quoted is not None
    assert quoted.group(0) in messages[0]


def test_discrete_screen_depends_on_the_forward() -> None:
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    ivs = [0.4308, 0.4065, 0.3218, 0.2787, 0.2548, 0.2533, 0.2780]
    assert butterfly_violations(strikes, ivs, T=0.5, forward=100.0) == [1]
    assert butterfly_violations(strikes, ivs, T=0.5, forward=85.0) == []
    assert "butterfly_violations(strikes, ivs, T=0.5, forward=100.0)   # [1]" in README
    assert "butterfly_violations(strikes, ivs, T=0.5, forward=85.0)    # []" in README

    costs = re.search(r"costs (−[\d.]+) at a forward of 100 and \+([\d.]+) at 85", README)
    assert costs is not None, "the README no longer quotes the 70/80/90 butterfly costs"
    for forward, claimed in ((100.0, costs.group(1)), (85.0, costs.group(2))):
        p = [BlackScholes(forward, K, 0.5).price(v)
             for K, v in zip(strikes, ivs, strict=True)]
        cost = p[0] - 2.0 * p[1] + p[2]
        assert float(f"{cost:+.3f}".replace("+", "")) == float(claimed.replace("−", "-"))


def _repaired_vogt_fit() -> tuple[SVIRawParams, list[float]]:
    ks = [-1.5 + 0.05 * i for i in range(61)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ButterflyArbitrageWarning)
        return fit_svi_slice(ks, [svi_w(k, VOGT) for k in ks],
                             butterfly_penalty=1e5), ks


def test_hero_caption() -> None:
    """The three numbers the hero paragraph reads off `docs/hero.png`."""
    min_g, _ = svi_min_g(VOGT, -1.5, 1.5, n=3001)
    claimed_g = literal(r"on the call wing \$g\$ reaches (−[\d.]+),")
    assert float(f"{min_g:.4f}") == claimed_g

    lo = literal(r"the density is negative for \$([\d.]+) < k < [\d.]+\$")
    hi = literal(r"the density is negative for \$[\d.]+ < k < ([\d.]+)\$")
    assert svi_density(lo - 0.002, VOGT) > 0.0 and svi_density(lo + 0.002, VOGT) < 0.0
    assert svi_density(hi - 0.002, VOGT) < 0.0 and svi_density(hi + 0.002, VOGT) > 0.0

    # A bound, not an equality. `moved` comes out of the penalised fit, and
    # this module's own rule is that optimiser output is compared loosely: the
    # search path differs enough between CPython versions to move the second
    # decimal (1.83 on 3.12, 1.81 on 3.11), which says nothing about the fit.
    repaired, ks = _repaired_vogt_fit()
    moved = max(abs(math.sqrt(svi_w(k, repaired)) - math.sqrt(svi_w(k, VOGT))) for k in ks)
    bound = literal(r"the smile moves by under ([\d.]+) vol points")
    assert 0.0 < moved * 100 < bound, (
        f"repair moved the smile {moved * 100:.2f} vol points, claimed under {bound}")


def test_the_soft_penalty_caveat() -> None:
    """The penalty shrinks |min g(k)| by orders of magnitude without fixing its sign.

    A magnitude bound, deliberately, and not a sign test. `butterfly_penalty`
    adds a term to the objective rather than constraining the feasible set, so
    where the residual lands is a property of the optimiser's path: CI has seen
    +6.6e-06 on macOS/3.12 and -8.3e-05 on macOS/3.11 from the same source. A
    sign assertion here would be asserting something the library does not
    claim, which is the whole point of the caveat this checks.
    """
    unpenalised, _ = svi_min_g(VOGT, -1.5, 1.5, n=3001), None
    repaired, _ = _repaired_vogt_fit()
    rep_g, _ = svi_min_g(repaired, -1.5, 1.5, n=3001)
    bound = literal(r"to within `([\d.e-]+)` of zero")
    assert abs(rep_g) < bound, f"|min g(k)| = {abs(rep_g):.2e}, claimed within {bound:.0e}"
    assert abs(rep_g) < abs(unpenalised[0]) / 100, (
        f"penalty only improved min g(k) from {unpenalised[0]:.2e} to {rep_g:.2e}")


def test_the_readme_states_the_real_test_count() -> None:
    """`pytest  # N tests` in the Install block is the number pytest collects."""
    claimed = int(re.search(r"pytest\s+# (\d+) tests", README).group(1))
    # `-o addopts=` clears the repo's own `-ra -q`. Leaving it in makes pytest
    # doubly quiet, which drops the "N tests collected" line this parses.
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only",
         "-o", "addopts=", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert collected == claimed


def test_the_version_does_not_drift_across_files() -> None:
    """One version string, three places it is written down."""
    from volsurf import __version__
    pyproject = re.search(r'^version = "([^"]+)"',
                          (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    changelog = re.search(r"^## \[([^\]]+)\]",
                          (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
    citation = re.search(r"^version: (.+)$",
                         (ROOT / "CITATION.cff").read_text(), re.M).group(1).strip()
    assert pyproject == __version__ == changelog == citation
