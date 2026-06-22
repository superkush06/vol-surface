"""Multi-expiry SVI surface tests."""

import pytest

from volsurf.surface import SVISurface, fit_svi_slice, fit_svi_surface
from volsurf.svi import SVIRawParams, svi_w

KS = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]


def _slice_w(params: SVIRawParams, ks=KS):
    return [svi_w(k, params) for k in ks]


def test_fit_svi_slice_recovers_known_params():
    truth = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    w = _slice_w(truth)
    fit = fit_svi_slice(KS, w, max_iter=1500)
    # reconstructed total variance should match closely at every k
    for k, w_true in zip(KS, w, strict=False):
        assert svi_w(k, fit) == pytest.approx(w_true, abs=1e-3)


def test_fit_svi_slice_rejects_short_input():
    with pytest.raises(ValueError):
        fit_svi_slice([0.0, 0.1], [0.04, 0.04])


def test_surface_interpolates_total_variance_in_T():
    p_short = SVIRawParams(a=0.02, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    p_long = SVIRawParams(a=0.06, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    surf = SVISurface(expiries=[0.5, 1.0], params=[p_short, p_long])
    # at a fitted expiry, returns that slice exactly
    assert surf.total_variance(0.0, 0.5) == pytest.approx(svi_w(0.0, p_short))
    # halfway between, linear interpolation of w
    mid = surf.total_variance(0.0, 0.75)
    assert mid == pytest.approx(0.5 * (svi_w(0.0, p_short) + svi_w(0.0, p_long)))


def test_surface_iv_positive_and_clamps_outside_range():
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    surf = SVISurface(expiries=[0.5, 1.5], params=[p, p])
    assert surf.iv(0.0, 0.25) > 0          # T below range -> clamp to first
    assert surf.iv(0.0, 3.0) > 0           # T above range -> clamp to last


def test_calendar_arbitrage_free_detects_good_and_bad():
    # increasing total variance with T -> arb-free
    lo = SVIRawParams(a=0.02, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    hi = SVIRawParams(a=0.08, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    assert SVISurface([0.5, 1.0], [lo, hi]).calendar_arbitrage_free()
    # inverted (variance falls with T) -> calendar arbitrage
    assert not SVISurface([0.5, 1.0], [hi, lo]).calendar_arbitrage_free()


def test_fit_svi_surface_end_to_end():
    truths = {
        0.25: SVIRawParams(a=0.02, b=0.10, rho=-0.3, m=0.0, sigma=0.2),
        1.00: SVIRawParams(a=0.06, b=0.12, rho=-0.3, m=0.0, sigma=0.2),
    }
    slices = [(T, KS, _slice_w(p)) for T, p in truths.items()]
    surf = fit_svi_surface(slices)
    assert surf.expiries == [0.25, 1.0]
    assert surf.calendar_arbitrage_free()
    # fitted IV close to truth at the money
    for T, p in truths.items():
        true_iv = (svi_w(0.0, p) / T) ** 0.5
        assert surf.iv(0.0, T) == pytest.approx(true_iv, abs=1e-2)
