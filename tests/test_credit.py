import numpy as np
import pytest

from src.config import COUNTERPARTIES_FILE
from src.credit import CounterpartyCredit, HazardCurve, credit_curve_table, hazard_from_spread, load_counterparties

GRID = np.linspace(0.0, 10.0, 121)


def test_flat_hazard_survival_is_exponential():
    curve = HazardCurve.flat(0.03)

    np.testing.assert_allclose(curve.survival(GRID), np.exp(-0.03 * GRID), rtol=1e-14)
    assert curve.survival(0.0) == pytest.approx(1.0)


def test_piecewise_hazard_integrates_segment_by_segment():
    curve = HazardCurve(np.array([1.0, 3.0, 5.0]), np.array([0.01, 0.02, 0.03]))

    assert curve.integrated_hazard(0.5) == pytest.approx(0.005)
    assert curve.integrated_hazard(2.0) == pytest.approx(0.01 + 0.02)
    assert curve.integrated_hazard(5.0) == pytest.approx(0.01 + 0.04 + 0.06)
    assert curve.integrated_hazard(6.0) == pytest.approx(0.01 + 0.04 + 0.06 + 0.03)  # last rate continues flat


@pytest.mark.parametrize("curve", [HazardCurve.flat(0.05), HazardCurve(np.array([2.0, 5.0]), np.array([0.01, 0.08]))])
def test_survival_falls_and_pd_rises(curve):
    assert np.all(np.diff(curve.survival(GRID)) <= 0)
    assert np.all(np.diff(curve.cumulative_pd(GRID)) >= 0)
    assert np.all(curve.marginal_pd(GRID) >= 0)


def test_marginal_pds_sum_to_cumulative_pd():
    curve = HazardCurve(np.array([2.0, 5.0]), np.array([0.01, 0.08]))

    assert curve.marginal_pd(GRID).sum() == pytest.approx(curve.cumulative_pd(GRID[-1]), rel=1e-12)


def test_zero_hazard_means_no_default():
    curve = HazardCurve.flat(0.0)

    np.testing.assert_array_equal(curve.survival(GRID), 1.0)
    np.testing.assert_array_equal(curve.marginal_pd(GRID), 0.0)


def test_credit_triangle():
    assert hazard_from_spread(0.015, 0.40) == pytest.approx(0.025)
    assert hazard_from_spread(0.030, 0.40) == pytest.approx(0.050)
    with pytest.raises(ValueError):
        hazard_from_spread(0.01, 1.0)


@pytest.mark.parametrize("times, hazards", [([1.0], [-0.01]), ([0.0], [0.01]), ([2.0, 1.0], [0.01, 0.02])])
def test_invalid_hazard_curves_raise(times, hazards):
    with pytest.raises(ValueError):
        HazardCurve(np.array(times), np.array(hazards))


def test_counterparty_file_loads_with_spreads_in_decimals():
    credits = load_counterparties(COUNTERPARTIES_FILE)

    assert set(credits) == {"NS_A", "NS_B"}
    assert credits["NS_A"] == CounterpartyCredit("CPTY_A", "NS_A", 0.015, 0.40)
    assert credits["NS_B"].hazard == pytest.approx(0.05)
    assert credits["NS_B"].lgd == pytest.approx(0.60)


def test_credit_curve_table_is_consistent():
    table = credit_curve_table(load_counterparties(COUNTERPARTIES_FILE), GRID)

    for _, rows in table.groupby("counterparty_id"):
        np.testing.assert_allclose(rows["survival"] + rows["cumulative_pd"], 1.0, rtol=1e-14)
        assert rows["marginal_pd"].sum() == pytest.approx(rows["cumulative_pd"].iloc[-1], rel=1e-12)
