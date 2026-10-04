from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from src.config import CURVE_FILE, HW_PARAMS, PFE_QUANTILE, PORTFOLIO_FILE
from src.curves import load_treasury_curve
from src.exposure import (
    expected_exposure,
    expected_positive_exposure,
    exposure_summary,
    gross_exposure,
    netted_exposure,
    netting_set_horizons,
    positive_exposure,
    potential_future_exposure,
    profile_table,
)
from src.instruments import load_portfolio
from src.rate_models import HullWhiteParams
from src.simulation import build_time_grid, netting_set_mtm, revalue_portfolio, simulate_paths
from src.swap_pricer import swap_mtm

Z_TOL = 4.0


@pytest.fixture(scope="module")
def curve():
    return load_treasury_curve(CURVE_FILE)


@pytest.fixture(scope="module")
def portfolio():
    return load_portfolio(PORTFOLIO_FILE)


@pytest.fixture(scope="module")
def grid(portfolio):
    return build_time_grid(portfolio)


@pytest.fixture(scope="module")
def cube(curve, portfolio, grid):
    paths = simulate_paths(HW_PARAMS, grid, 4_000, np.random.default_rng(21))
    return revalue_portfolio(portfolio, curve, paths)


def test_exposure_non_negative(cube, portfolio):
    exposure = positive_exposure(cube)

    assert np.all(exposure >= 0.0)
    np.testing.assert_array_equal(exposure[cube > 0], cube[cube > 0])
    for netted in netted_exposure(cube, portfolio).values():
        assert np.all(netted >= 0.0)


def test_netted_not_above_gross(cube, portfolio):
    # max(sum V, 0) <= sum max(V, 0) on every path and date (spec section 11)
    netted, gross = netted_exposure(cube, portfolio), gross_exposure(cube, portfolio)

    for ns in netted:
        assert np.all(netted[ns] <= gross[ns] + 1e-9)


def test_netting_set_with_one_trade_has_no_netting_benefit(cube, portfolio):
    single = [replace(swap, netting_set_id=f"SOLO_{swap.trade_id}") for swap in portfolio]
    netted, gross = netted_exposure(cube, single), gross_exposure(cube, single)

    for ns in netted:
        np.testing.assert_array_equal(netted[ns], gross[ns])


def test_gross_exposure_is_sum_of_trade_exposures(cube, portfolio):
    gross = gross_exposure(cube, portfolio)
    members = [i for i, swap in enumerate(portfolio) if swap.netting_set_id == "NS_A"]

    np.testing.assert_allclose(gross["NS_A"], np.maximum(cube[members], 0).sum(axis=0), rtol=1e-12)


def test_ee_at_time_zero_is_positive_part_of_todays_value(curve, cube, portfolio):
    netted = netted_exposure(cube, portfolio)
    for ns in netted:
        value_today = sum(swap_mtm(swap, curve) for swap in portfolio if swap.netting_set_id == ns)
        assert expected_exposure(netted[ns])[0] == pytest.approx(max(value_today, 0.0), rel=1e-10)


def test_pfe_reconciles_with_simulated_distribution(cube, portfolio):
    # PFE_q is the smallest simulated exposure with empirical CDF >= q
    for exposure in netted_exposure(cube, portfolio).values():
        pfe = potential_future_exposure(exposure, PFE_QUANTILE)
        cdf_at = (exposure <= pfe).mean(axis=0)
        cdf_below = (exposure < pfe).mean(axis=0)

        assert np.all(cdf_at >= PFE_QUANTILE)
        assert np.all(cdf_below < PFE_QUANTILE)
        assert np.all(np.any(np.isclose(exposure, pfe), axis=0))  # an actual simulated value


def test_ee_and_pfe_match_a_known_distribution():
    # V ~ N(0, 1): EE = E[max(Z, 0)] = 1/sqrt(2 pi) and PFE_q = Phi^{-1}(q)
    n = 200_000
    exposure = positive_exposure(np.random.default_rng(3).standard_normal((n, 1)))
    ee_se = exposure.std() / np.sqrt(n)
    q = PFE_QUANTILE
    pfe_se = np.sqrt(q * (1 - q) / n) / norm.pdf(norm.ppf(q))

    assert abs(expected_exposure(exposure)[0] - 1 / np.sqrt(2 * np.pi)) <= Z_TOL * ee_se
    assert abs(potential_future_exposure(exposure, q)[0] - norm.ppf(q)) <= Z_TOL * pfe_se


def test_pfe_rises_with_quantile(cube, portfolio):
    exposure = netted_exposure(cube, portfolio)["NS_B"]

    assert np.all(potential_future_exposure(exposure, 0.99) >= potential_future_exposure(exposure, 0.95))
    assert np.all(potential_future_exposure(exposure, 0.95) >= potential_future_exposure(exposure, 0.50))


@pytest.mark.parametrize("quantile", [0.0, 1.0, 1.5])
def test_invalid_quantile_raises(quantile):
    with pytest.raises(ValueError):
        potential_future_exposure(np.ones((10, 2)), quantile)


def test_epe_is_the_time_average_of_ee():
    times = np.array([0.0, 0.5, 1.0, 2.0])

    assert expected_positive_exposure(times, np.full(4, 3.0)) == pytest.approx(3.0)
    assert expected_positive_exposure(times, times / 2.0) == pytest.approx(0.5)  # EE = t/2 on [0, 2]
    assert expected_positive_exposure(times, times, horizon=1.0) == pytest.approx(0.5)


def test_zero_volatility_exposure_is_positive_part_of_forward_value(curve, portfolio, grid):
    flat = HullWhiteParams(HW_PARAMS.mean_reversion, 0.0)
    cube = revalue_portfolio(portfolio, curve, simulate_paths(flat, grid, 3, np.random.default_rng(0)))
    values = netting_set_mtm(cube, portfolio)
    netted = netted_exposure(cube, portfolio)

    for ns in netted:
        forward_value = values[ns][0]
        np.testing.assert_allclose(expected_exposure(netted[ns]), np.maximum(forward_value, 0.0), atol=1e-6)
        np.testing.assert_allclose(potential_future_exposure(netted[ns]), np.maximum(forward_value, 0.0), atol=1e-6)


def test_exposure_is_zero_once_every_trade_has_matured(cube, portfolio, grid):
    horizons = netting_set_horizons(portfolio)
    for ns, exposure in netted_exposure(cube, portfolio).items():
        assert np.all(exposure[:, grid >= horizons[ns] - 1e-9] == 0.0)


def test_summary_and_profile_tables_are_consistent(cube, portfolio, grid):
    netted, gross = netted_exposure(cube, portfolio), gross_exposure(cube, portfolio)
    summary = exposure_summary(grid, netted, gross, netting_set_horizons(portfolio)).set_index("netting_set_id")
    profiles = profile_table(grid, netted, gross)

    for ns in netted:
        rows = profiles[profiles["netting_set_id"] == ns]
        assert summary.loc[ns, "peak_ee_netted"] == pytest.approx(rows["ee_netted"].max())
        assert summary.loc[ns, "epe_netted"] <= summary.loc[ns, "epe_gross"] + 1e-9
        assert 0.0 <= summary.loc[ns, "netting_reduction"] < 1.0
    assert summary.loc["NS_A", "netting_reduction"] > summary.loc["NS_B", "netting_reduction"]
