"""Implied volatility tests."""

import math

import pytest

from volsurf.black_scholes import BlackScholes
from volsurf.iv import IVSolverError, implied_vol


def test_iv_roundtrips_for_call():
    bs = BlackScholes(S=100, K=105, T=0.5, r=0.02, q=0.0)
    true_sigma = 0.25
    price = bs.price(sigma=true_sigma, call=True)
    iv = implied_vol(price, bs, call=True)
    assert iv == pytest.approx(true_sigma, abs=1e-6)


def test_iv_roundtrips_for_put():
    bs = BlackScholes(S=100, K=95, T=1.5, r=0.01)
    true_sigma = 0.35
    price = bs.price(sigma=true_sigma, call=False)
    iv = implied_vol(price, bs, call=False)
    assert iv == pytest.approx(true_sigma, abs=1e-6)


def test_iv_rejects_below_intrinsic():
    bs = BlackScholes(S=120, K=100, T=0.5)
    with pytest.raises(IVSolverError):
        implied_vol(market_price=5.0, bs=bs, call=True)


def test_iv_error_names_upper_bracket_when_vol_exceeds_hi():
    """A 600% vol (short-dated event pricing) is above the default hi=5.

    The solver must say the bracket is too small — the old message was a
    bare 'no sign change in [1e-06, 5.0]' — and a larger hi must solve it.
    """
    bs = BlackScholes(S=100, K=100, T=0.01)
    price = bs.price(sigma=6.0)
    with pytest.raises(IVSolverError, match="exceeds the upper bracket"):
        implied_vol(price, bs)
    assert implied_vol(price, bs, hi=10.0) == pytest.approx(6.0, abs=1e-6)


def test_iv_error_names_lower_bracket_when_vol_below_lo():
    bs = BlackScholes(S=100, K=100, T=1.0)
    price = bs.price(sigma=0.2)
    with pytest.raises(IVSolverError, match="below the lower bracket"):
        implied_vol(price, bs, lo=0.5)


def test_iv_handles_deep_itm_call():
    bs = BlackScholes(S=120, K=100, T=0.5, r=0.03)
    price = bs.price(sigma=0.3, call=True)
    iv = implied_vol(price, bs, call=True)
    assert iv == pytest.approx(0.3, abs=1e-5)


def test_iv_returns_the_bracket_end_when_the_price_sits_on_it():
    """Deep out of the money the price is flat in sigma to the last bit.

    Then price(lo) == market_price exactly, f(lo) is +0.0, and there is no
    sign change for Brent to work with — the solver used to report the
    internal "f(a)*f(b) >= 0" instead of the perfectly good root it already
    had in hand.
    """
    bs = BlackScholes(S=100.0, K=100.0 * math.exp(6.0), T=0.02)
    price = bs.price(sigma=1e-6)
    assert price == bs.price(sigma=2e-6)   # flat: the bracket end is the root
    assert implied_vol(price, bs) == 1e-6
