"""Black-Scholes tests."""

import math

import pytest

from volsurf.black_scholes import BlackScholes


def test_atm_call_price_positive():
    bs = BlackScholes(S=100, K=100, T=1.0, r=0.0, q=0.0)
    p = bs.price(sigma=0.2, call=True)
    assert p > 0


def test_put_call_parity():
    bs = BlackScholes(S=100, K=110, T=0.5, r=0.03, q=0.01)
    c = bs.price(sigma=0.25, call=True)
    p = bs.price(sigma=0.25, call=False)
    parity_rhs = bs.S * math.exp(-bs.q * bs.T) - bs.K * math.exp(-bs.r * bs.T)
    assert (c - p) == pytest.approx(parity_rhs, abs=1e-9)


def test_vega_positive():
    bs = BlackScholes(S=100, K=100, T=1.0, r=0.0, q=0.0)
    assert bs.vega(0.2) > 0


def test_call_delta_in_unit_interval():
    bs = BlackScholes(S=100, K=100, T=1.0, r=0.0, q=0.0)
    d = bs.delta(0.2, call=True)
    assert 0.0 < d < 1.0


def test_zero_expiry_is_intrinsic():
    bs = BlackScholes(S=120, K=100, T=0.0)
    assert bs.price(sigma=0.2, call=True) == 20.0
    assert bs.price(sigma=0.2, call=False) == 0.0
