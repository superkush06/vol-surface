"""SVI tests."""

import math

import pytest

from volsurf.black_scholes import BlackScholes
from volsurf.svi import (
    SVIRawParams,
    svi_butterfly_arbitrage_free,
    svi_density,
    svi_g,
    svi_iv,
    svi_min_g,
    svi_w,
)

# the Axel Vogt parameters from Gatheral & Jacquier (2014), the classic
# example of an SVI slice that admits butterfly arbitrage
VOGT = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_svi_w_positive_in_typical_range():
    p = SVIRawParams(a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.4)
    for k in [-0.5, -0.1, 0.0, 0.1, 0.5]:
        assert svi_w(k, p) > 0


def test_svi_validates():
    with pytest.raises(ValueError):
        SVIRawParams(a=0.01, b=-0.1, rho=0.0, m=0.0, sigma=0.4).validate()
    with pytest.raises(ValueError):
        SVIRawParams(a=0.01, b=0.1, rho=1.5, m=0.0, sigma=0.4).validate()
    with pytest.raises(ValueError):
        SVIRawParams(a=0.01, b=0.1, rho=0.0, m=0.0, sigma=-0.1).validate()


def test_svi_iv_roundtrip_w():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.3)
    T = 0.5
    iv = svi_iv(0.0, T, p)
    assert iv ** 2 * T == pytest.approx(svi_w(0.0, p))


def test_svi_g_is_one_for_flat_slice():
    """b = 0 gives constant w: g(k) = 1 exactly and the density is the
    lognormal density of k."""
    flat = SVIRawParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    for k in (-0.5, 0.0, 0.3, 1.0):
        assert svi_g(k, flat) == pytest.approx(1.0, abs=1e-15)
    w = 0.04
    k = 0.3
    d_minus = -k / math.sqrt(w) - math.sqrt(w) / 2.0
    ref = math.exp(-0.5 * d_minus * d_minus) / math.sqrt(2.0 * math.pi * w)
    assert svi_density(k, flat) == pytest.approx(ref, rel=1e-14)


def test_svi_g_positive_for_arbitrage_free_params():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    assert svi_butterfly_arbitrage_free(p, -1.5, 1.5)
    for i in range(61):
        assert svi_g(-1.5 + 0.05 * i, p) > 0


def test_svi_g_detects_vogt_butterfly_arbitrage():
    """Gatheral-Jacquier's textbook arbitrageable slice: g dips below zero
    around k ~ 0.9 and the implied density goes negative there."""
    assert not svi_butterfly_arbitrage_free(VOGT, -1.5, 1.5)
    min_k = min((i / 100.0 for i in range(-150, 151)),
                key=lambda k: svi_g(k, VOGT))
    assert svi_g(min_k, VOGT) < 0
    assert svi_density(min_k, VOGT) < 0
    assert 0.5 < min_k < 1.2


def test_svi_density_integrates_to_one():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    lo, hi, n = -8.0, 8.0, 4001
    h = (hi - lo) / (n - 1)
    total = sum(svi_density(lo + i * h, p) * (h if 0 < i < n - 1 else h / 2)
                for i in range(n))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_svi_density_matches_finite_difference_of_call_prices():
    """p(k) must equal K * d2C/dK2 with C priced at forward F = 1."""
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    T = 1.0

    def call(K):
        k = math.log(K)
        return BlackScholes(1.0, K, T).price(math.sqrt(svi_w(k, p) / T))

    for k in (-0.4, 0.0, 0.5):
        K = math.exp(k)
        h = 1e-4 * K
        fd = (call(K - h) - 2.0 * call(K) + call(K + h)) / (h * h)
        assert svi_density(k, p) == pytest.approx(fd * K, rel=1e-5)


def test_svi_butterfly_arbitrage_free_validates_range():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        svi_butterfly_arbitrage_free(p, 1.0, -1.0)


def test_svi_min_g_returns_value_and_location():
    """min g and its argument must agree with a brute-force scan on the
    same grid, and with the boolean check derived from it."""
    min_g, k_star = svi_min_g(VOGT, -1.5, 1.5, n=601)
    grid = [-1.5 + 3.0 * i / 600.0 for i in range(601)]
    assert min_g == pytest.approx(min(svi_g(k, VOGT) for k in grid))
    assert svi_g(k_star, VOGT) == pytest.approx(min_g)
    assert 0.5 < k_star < 1.2
    assert not svi_butterfly_arbitrage_free(VOGT, -1.5, 1.5, n=601)
    # a tolerance as loose as the violation makes the same slice "pass"
    assert svi_butterfly_arbitrage_free(VOGT, -1.5, 1.5, n=601, tol=0.05)


def test_svi_min_g_validates_grid():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        svi_min_g(p, 1.0, -1.0)
    with pytest.raises(ValueError):
        svi_min_g(p, -1.0, 1.0, n=1)
