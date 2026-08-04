"""SABR tests."""


import pytest

from volsurf.sabr import SABRParams, sabr_iv
from volsurf.sabr_fit import fit_sabr


def test_atm_iv_matches_alpha_when_beta_one():
    p = SABRParams(alpha=0.3, beta=1.0, rho=0.0, nu=0.5)
    iv_atm = sabr_iv(F=100.0, K=100.0, T=1.0, params=p)
    assert iv_atm == pytest.approx(0.3, rel=0.1)


def test_atm_iv_equals_hagan_atm_closed_form():
    """ATM through the general code path must equal Hagan's ATM formula."""
    import math
    alpha, beta, rho, nu = 0.3, 0.5, -0.4, 0.6
    F, T = 100.0, 1.0
    p = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    omb = 1.0 - beta
    expected = (alpha / (F ** omb)) * (1.0 + (
        ((omb * alpha) ** 2) / (24.0 * (F ** (2.0 * omb)))
        + 0.25 * rho * beta * nu * alpha / (F ** omb)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * T)
    assert math.isclose(sabr_iv(F, F, T, p), expected, rel_tol=1e-14)


def test_near_atm_converges_monotonically_no_spike():
    """K -> F sweep: |iv(K) - iv(F)| must shrink monotonically to ~0.

    The old near-ATM branch returned the leading factor without the
    (1 + [...]T) time correction, producing a ~2% vol spike in a band
    around K = F + 2e-12.
    """
    p = SABRParams(alpha=0.3, beta=0.5, rho=-0.4, nu=0.6)
    F, T = 100.0, 1.0
    atm = sabr_iv(F, F, T, p)
    diffs = [abs(sabr_iv(F, F * (1.0 + 10.0 ** -pw), T, p) - atm)
             for pw in range(3, 15)]
    for lo, hi in zip(diffs[1:], diffs[:-1], strict=True):
        assert lo <= hi + 1e-15
    assert diffs[-1] < 1e-10


def test_no_discontinuity_band_just_off_the_money():
    """The exact offsets from the audit of the old code: F+1e-12 vs F+2e-12."""
    p = SABRParams(alpha=0.3, beta=0.5, rho=-0.4, nu=0.6)
    F, T = 100.0, 1.0
    atm = sabr_iv(F, F, T, p)
    for dk in (1e-12, 2e-12, 5e-12):
        assert sabr_iv(F, F + dk, T, p) == pytest.approx(atm, abs=1e-9)


def test_golden_values_pinned():
    """Regression anchors for the Hagan transcription (away from ATM the
    formula is unchanged; ATM now flows through the same path)."""
    p = SABRParams(alpha=0.3, beta=0.5, rho=-0.4, nu=0.6)
    assert sabr_iv(100.0, 80.0, 1.0, p) == pytest.approx(0.07208623291660365, rel=1e-12)
    assert sabr_iv(100.0, 100.0, 1.0, p) == pytest.approx(0.030657281249999994, rel=1e-12)
    assert sabr_iv(100.0, 120.0, 1.0, p) == pytest.approx(0.045612368989418554, rel=1e-12)
    p2 = SABRParams(alpha=0.2, beta=1.0, rho=0.3, nu=0.4)
    assert sabr_iv(100.0, 90.0, 0.5, p2) == pytest.approx(0.196762512579204, rel=1e-12)


def test_iv_smile_shape_for_negative_rho():
    p = SABRParams(alpha=0.3, beta=1.0, rho=-0.5, nu=0.5)
    F, T = 100.0, 1.0
    iv_lo = sabr_iv(F, 80.0, T, p)
    iv_hi = sabr_iv(F, 120.0, T, p)
    # Negative skew: low strikes have higher IV
    assert iv_lo > iv_hi


def test_params_validate():
    with pytest.raises(ValueError):
        SABRParams(alpha=-0.1, beta=0.5, rho=0.0, nu=0.5).validate()
    with pytest.raises(ValueError):
        SABRParams(alpha=0.3, beta=1.5, rho=0.0, nu=0.5).validate()
    with pytest.raises(ValueError):
        SABRParams(alpha=0.3, beta=0.5, rho=1.5, nu=0.5).validate()


def test_fit_sabr_recovers_synthetic():
    """Generate synthetic IVs from a known SABR; fit; check we recover."""
    truth = SABRParams(alpha=0.25, beta=0.5, rho=-0.3, nu=0.6)
    F, T = 100.0, 1.0
    strikes = [80, 90, 95, 100, 105, 110, 120]
    market_ivs = [sabr_iv(F, K, T, truth) for K in strikes]
    fit = fit_sabr(F, T, strikes, market_ivs, beta=0.5, max_iter=400)
    # Reconstruct IVs from fit, check close
    for K, target in zip(strikes, market_ivs, strict=False):
        got = sabr_iv(F, K, T, fit)
        assert got == pytest.approx(target, abs=0.01)
