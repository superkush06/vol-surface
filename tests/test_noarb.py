"""No-arbitrage check tests."""

import pytest

from volsurf.black_scholes import BlackScholes
from volsurf.noarb import (
    butterfly_violations,
    calendar_violations,
    total_variance,
)


def test_butterfly_no_violations_on_constant_iv():
    strikes = [80, 90, 100, 110, 120]
    ivs = [0.2] * 5
    assert butterfly_violations(strikes, ivs, T=1.0, forward=100.0) == []


def test_butterfly_detects_violation_missed_at_wrong_spot():
    """A genuine butterfly the old pay-at-middle-strike scan waved through.

    At the true forward the 70/80/90 butterfly costs -0.392 (free money);
    priced with S set to the middle strike of each triple, the same smile
    looks clean. The check must price every leg at the actual forward.
    """
    strikes = [70, 80, 90, 100, 110, 120, 130]
    ivs = [0.4308, 0.4065, 0.3218, 0.2787, 0.2548, 0.2533, 0.278]
    F, T = 100.0, 0.5
    # the butterfly spread really is negative at the true forward
    def call(K, iv):
        return BlackScholes(F, K, T).price(iv)
    cost = call(70, ivs[0]) - 2.0 * call(80, ivs[1]) + call(90, ivs[2])
    assert cost < 0
    assert butterfly_violations(strikes, ivs, T, forward=F) == [1]


def test_butterfly_rejects_bad_inputs():
    with pytest.raises(ValueError):
        butterfly_violations([90, 100, 110], [0.2, 0.2], T=1.0, forward=100.0)
    with pytest.raises(ValueError):
        butterfly_violations([90, 100, 110], [0.2] * 3, T=1.0, forward=-1.0)


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


def test_calendar_rejects_mismatched_lengths():
    """Mismatched IV lists must raise, not silently check a truncated set.

    The old zip(strict=False) scanned only 1 of 3 strikes here and
    returned as if the other two had been checked.
    """
    with pytest.raises(ValueError):
        calendar_violations([90, 100, 110], [0.2] * 3, [0.1],
                            T_short=0.5, T_long=1.0)
    with pytest.raises(ValueError):
        calendar_violations([90, 100, 110], [0.2], [0.1] * 3,
                            T_short=0.5, T_long=1.0)


def test_total_variance_grows_with_t():
    assert total_variance(0.2, 1.0) < total_variance(0.2, 2.0)
