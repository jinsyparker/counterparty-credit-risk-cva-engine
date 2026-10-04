import numpy as np
import pytest

from src.config import CURVE_FILE
from src.curves import DiscountCurve, bootstrap_par_yields, load_par_yields, load_treasury_curve, par_bond_schedule


@pytest.fixture(scope="module")
def treasury_curve():
    return load_treasury_curve(CURVE_FILE)


def test_curve_reproduces_input_nodes():
    times = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    zeros = np.array([0.040, 0.042, 0.045, 0.047, 0.050])
    curve = DiscountCurve(times, zeros)

    np.testing.assert_allclose(curve.zero_rate(times), zeros, rtol=0, atol=1e-14)
    np.testing.assert_allclose(curve.df(times), np.exp(-zeros * times), rtol=0, atol=1e-14)


def test_discount_factor_is_one_at_time_zero(treasury_curve):
    assert treasury_curve.df(0.0) == pytest.approx(1.0, abs=1e-15)


def test_discount_factors_positive_and_decreasing(treasury_curve):
    grid = np.linspace(0.0, 40.0, 4001)
    dfs = treasury_curve.df(grid)

    assert np.all(dfs > 0)
    assert np.all(np.diff(dfs) < 0)


def test_bootstrap_reprices_every_input_par_bond(treasury_curve):
    quotes = load_par_yields(CURVE_FILE)
    for tenor, par_yield in zip(quotes["tenor_years"], quotes["par_yield"]):
        pay_times, accruals = par_bond_schedule(tenor)
        price = par_yield * np.sum(accruals * treasury_curve.df(pay_times)) + treasury_curve.df(tenor)
        assert price == pytest.approx(1.0, abs=1e-10), f"{tenor}Y bond does not reprice to par"


def test_flat_par_curve_bootstraps_to_flat_zero_curve():
    # A flat 5% semiannual par curve is a flat 2*ln(1.025) continuous zero curve beyond 6M
    tenors = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    curve = bootstrap_par_yields(tenors, np.full(tenors.size, 0.05))

    expected = 2.0 * np.log(1.025)
    np.testing.assert_allclose(curve.zero_rate(np.array([1.0, 3.0, 7.5, 10.0])), expected, atol=1e-12)


def test_forward_rate_on_flat_curve_equals_zero_rate():
    curve = DiscountCurve(np.array([1.0, 5.0, 10.0]), np.full(3, 0.04))

    assert curve.forward_rate(2.0, 7.0) == pytest.approx(0.04, abs=1e-14)
    assert curve.inst_forward(3.3) == pytest.approx(0.04, abs=1e-14)
    assert curve.simple_forward(1.0, 1.5) == pytest.approx((np.exp(0.04 * 0.5) - 1) / 0.5, abs=1e-14)


def test_extrapolation_holds_last_forward_flat():
    curve = DiscountCurve(np.array([1.0, 2.0]), np.array([0.03, 0.04]))
    last_forward = curve.forward_rate(1.0, 2.0)

    assert curve.forward_rate(2.0, 10.0) == pytest.approx(last_forward, abs=1e-14)
    assert curve.inst_forward(25.0) == pytest.approx(last_forward, abs=1e-14)


@pytest.mark.parametrize(
    "times, zeros",
    [([1.0, 1.0], [0.03, 0.04]), ([0.0, 1.0], [0.03, 0.04]), ([2.0, 1.0], [0.03, 0.04]), ([1.0], [0.03, 0.04])],
)
def test_invalid_curve_inputs_raise(times, zeros):
    with pytest.raises(ValueError):
        DiscountCurve(np.array(times), np.array(zeros))
