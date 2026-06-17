"""SVI tests."""

import pytest

from volsurf.svi import SVIRawParams, svi_iv, svi_w


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
