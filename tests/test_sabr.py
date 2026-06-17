"""SABR tests."""


import pytest

from volsurf.sabr import SABRParams, sabr_iv
from volsurf.sabr_fit import fit_sabr


def test_atm_iv_matches_alpha_when_beta_one():
    p = SABRParams(alpha=0.3, beta=1.0, rho=0.0, nu=0.5)
    iv_atm = sabr_iv(F=100.0, K=100.0, T=1.0, params=p)
    assert iv_atm == pytest.approx(0.3, rel=0.1)


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
