"""Implied volatility tests."""

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


def test_iv_handles_deep_itm_call():
    bs = BlackScholes(S=120, K=100, T=0.5, r=0.03)
    price = bs.price(sigma=0.3, call=True)
    iv = implied_vol(price, bs, call=True)
    assert iv == pytest.approx(0.3, abs=1e-5)
