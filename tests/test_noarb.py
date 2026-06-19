"""No-arbitrage check tests."""

from volsurf.noarb import (
    butterfly_violations,
    calendar_violations,
    total_variance,
)


def test_butterfly_no_violations_on_constant_iv():
    strikes = [80, 90, 100, 110, 120]
    ivs = [0.2] * 5
    assert butterfly_violations(strikes, ivs, T=1.0) == []


def test_calendar_no_violations_when_long_higher_variance():
    strikes = [90, 100, 110]
    ivs_short = [0.2] * 3
    ivs_long = [0.3] * 3
    out = calendar_violations(strikes, ivs_short, ivs_long, T_short=0.5, T_long=1.0)
    assert out == []


def test_calendar_detects_violation():
    strikes = [100]
    ivs_short = [0.5]
    ivs_long = [0.2]
    out = calendar_violations(strikes, ivs_short, ivs_long, T_short=0.5, T_long=1.0)
    assert out == [0]


def test_total_variance_grows_with_t():
    assert total_variance(0.2, 1.0) < total_variance(0.2, 2.0)
