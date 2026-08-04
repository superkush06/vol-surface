"""Multi-expiry SVI surface tests."""

import warnings

import pytest

from volsurf.surface import SVISurface, fit_svi_slice, fit_svi_surface
from volsurf.svi import (
    ButterflyArbitrageWarning,
    SVIRawParams,
    svi_butterfly_arbitrage_free,
    svi_g,
    svi_w,
)

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


def test_fit_svi_slice_recovers_steep_skew():
    """Exact data from a realistic steep equity skew must be recovered.

    The old single-start 5-D Nelder-Mead converged to a degenerate
    near-kink solution here (~19% max node error, right wing off 16x
    at k=2); the quasi-explicit reduction recovers every parameter to
    well under 0.1%.
    """
    truth = SVIRawParams(a=0.01, b=0.35, rho=-0.7, m=0.05, sigma=0.15)
    ks = [-0.5 + 0.05 * i for i in range(21)]
    w = [svi_w(k, truth) for k in ks]
    fit = fit_svi_slice(ks, w, max_iter=1500)
    assert fit.a == pytest.approx(truth.a, rel=1e-3, abs=1e-5)
    assert fit.b == pytest.approx(truth.b, rel=1e-3)
    assert fit.rho == pytest.approx(truth.rho, rel=1e-3)
    assert fit.m == pytest.approx(truth.m, rel=1e-3)
    assert fit.sigma == pytest.approx(truth.sigma, rel=1e-3)
    for k, w_true in zip(ks, w, strict=True):
        assert svi_w(k, fit) == pytest.approx(w_true, rel=1e-4)
    # extrapolated wing no longer blows up
    assert svi_w(2.0, fit) == pytest.approx(svi_w(2.0, truth), rel=1e-3)


def test_fit_svi_slice_rejects_short_input():
    with pytest.raises(ValueError):
        fit_svi_slice([0.0, 0.1], [0.04, 0.04])


def test_fit_warns_on_butterfly_arbitrageable_fit():
    """Fitting data from the Vogt slice recovers it — and must warn that
    the result admits butterfly arbitrage instead of returning silently."""
    vogt = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    ks = [-1.5 + 0.05 * i for i in range(61)]
    ws = [svi_w(k, vogt) for k in ks]
    with pytest.warns(ButterflyArbitrageWarning):
        fit = fit_svi_slice(ks, ws)
    assert not svi_butterfly_arbitrage_free(fit, -1.5, 1.5)


def test_fit_no_butterfly_warning_on_clean_smile():
    truth = SVIRawParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ButterflyArbitrageWarning)
        fit_svi_slice(KS, _slice_w(truth))


def test_butterfly_penalty_pushes_fit_toward_arbitrage_free():
    vogt = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    ks = [-1.5 + 0.05 * i for i in range(61)]
    ws = [svi_w(k, vogt) for k in ks]
    grid = [-1.5 + 3.0 * i / 200.0 for i in range(201)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ButterflyArbitrageWarning)
        plain = fit_svi_slice(ks, ws)
        penal = fit_svi_slice(ks, ws, butterfly_penalty=1.0)
    min_g_plain = min(svi_g(k, plain) for k in grid)
    min_g_penal = min(svi_g(k, penal) for k in grid)
    assert min_g_plain < -0.02          # the unpenalised fit is arbitrageable
    assert min_g_penal > min_g_plain + 0.02  # the penalty shrinks the violation


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
    assert surf.iv(0.0, 0.25) > 0          # T below range -> scaled first slice
    assert surf.iv(0.0, 3.0) > 0           # T above range -> clamp to last


def test_surface_short_end_iv_stays_flat_not_divergent():
    """Below the first expiry, total variance scales with T so implied vol
    stays at the first slice's level. The old constant-w clamp made
    iv(0, 0.001) = sqrt(w1/T) blow up to 775% vol."""
    p = SVIRawParams(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.2)
    surf = SVISurface(expiries=[0.5, 1.5], params=[p, p])
    iv_first = surf.iv(0.0, 0.5)
    assert surf.iv(0.0, 0.001) == pytest.approx(iv_first, rel=1e-12)
    assert surf.iv(0.0, 0.25) == pytest.approx(iv_first, rel=1e-12)
    # total variance still non-decreasing in T through the short end
    assert surf.total_variance(0.0, 0.1) < surf.total_variance(0.0, 0.5)
    assert surf.total_variance(0.0, 0.0) == 0.0


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


def test_surface_butterfly_check_is_per_slice():
    """A surface is only usable if every slice is admissible: one bad
    expiry is enough to fail, even when the calendar ordering is fine."""
    good = SVIRawParams(a=0.04, b=0.10, rho=-0.30, m=0.0000, sigma=0.20)
    vogt = SVIRawParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    clean = SVISurface([0.5, 1.0], [good, SVIRawParams(a=0.09, b=0.10,
                                                       rho=-0.30, m=0.0,
                                                       sigma=0.20)])
    assert clean.butterfly_arbitrage_free(-1.0, 1.0)
    assert clean.calendar_arbitrage_free()

    tainted = SVISurface([0.5, 1.0], [vogt, good])
    assert not tainted.butterfly_arbitrage_free(-1.5, 1.5)
