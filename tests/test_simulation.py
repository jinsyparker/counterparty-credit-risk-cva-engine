from dataclasses import replace

import numpy as np
import pytest

from src.config import CURVE_FILE, HW_PARAMS, PORTFOLIO_FILE
from src.curves import load_treasury_curve
from src.instruments import load_portfolio
from src.rate_models import HullWhiteParams
from src.simulation import build_time_grid, netting_set_mtm, revalue_portfolio, simulate_paths, swap_dates, time_index
from src.swap_pricer import cashflow_table, swap_mtm

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
def paths(grid):
    return simulate_paths(HW_PARAMS, grid, 20_000, np.random.default_rng(11))


def trade(portfolio, trade_id):
    return next(swap for swap in portfolio if swap.trade_id == trade_id)


def remaining_pv(swap, curve, t):
    """Time-0 PV of the cash flows paid after t (what E[D(0,t) V(t)] must equal)."""
    flows = cashflow_table(swap, curve)
    return flows.loc[flows["pay_time"] > t + 1e-9, "pv"].sum()


def test_grid_contains_every_portfolio_date_and_monthly_point(portfolio, grid):
    all_dates = np.concatenate([swap_dates(swap) for swap in portfolio] + [np.arange(121) / 12])

    time_index(grid, all_dates)  # raises if any date is missing
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(max(swap.maturity for swap in portfolio))
    assert np.all(np.diff(grid) > 0)


def test_grid_includes_extra_dates(portfolio):
    grid = build_time_grid(portfolio, extra_dates=np.array([0.0417, 3.3]))

    time_index(grid, np.array([0.0417, 3.3]))


def test_time_index_rejects_dates_off_the_grid(grid):
    with pytest.raises(ValueError):
        time_index(grid, 0.123)


def test_cube_at_time_zero_equals_time_zero_pricer(curve, portfolio, grid):
    sim = simulate_paths(HW_PARAMS, grid, 500, np.random.default_rng(1))
    cube = revalue_portfolio(portfolio, curve, sim)
    expected = np.array([swap_mtm(swap, curve) for swap in portfolio])

    assert cube.shape == (len(portfolio), 500, grid.size)
    np.testing.assert_allclose(cube[:, :, 0], np.broadcast_to(expected[:, None], (len(portfolio), 500)), rtol=1e-10, atol=1e-6)


def test_trades_are_worth_zero_from_their_final_payment(curve, portfolio, grid):
    sim = simulate_paths(HW_PARAMS, grid, 200, np.random.default_rng(1))
    cube = revalue_portfolio(portfolio, curve, sim)

    for i, swap in enumerate(portfolio):
        assert np.all(cube[i][:, grid >= swap.maturity - 1e-9] == 0.0)
        assert np.any(cube[i][:, grid < swap.maturity - 1e-9] != 0.0)


@pytest.mark.parametrize("t", [1.0, 2.0 + 1 / 12, 2.25, 5.5, 9.0 + 2 / 12])
def test_zero_volatility_values_equal_forward_values(curve, portfolio, grid, t):
    # sigma = 0: every path is the forward curve, so V(t) = sum of remaining PVs / P(0, t).
    # Mid-period dates check the stored fixing of the running floating coupon.
    flat = HullWhiteParams(HW_PARAMS.mean_reversion, 0.0)
    sim = simulate_paths(flat, grid, 2, np.random.default_rng(0))
    cube = revalue_portfolio(portfolio, curve, sim)
    k = time_index(grid, t)[0]

    for i, swap in enumerate(portfolio):
        expected = remaining_pv(swap, curve, t) / curve.df(t) if t < swap.maturity else 0.0
        np.testing.assert_allclose(cube[i, :, k], expected, rtol=1e-9, atol=1e-6)


def test_missing_reset_date_raises(curve, portfolio):
    swap = trade(portfolio, "A5")
    coarse = np.array([0.0, 0.4, 1.0, 2.0, 3.0])  # quarterly resets 0.25, 0.5, ... are missing
    sim = simulate_paths(HW_PARAMS, coarse, 10, np.random.default_rng(0))

    with pytest.raises(ValueError, match="reset date"):
        revalue_portfolio([swap], curve, sim)


def test_forward_starting_swap_value_is_a_martingale(curve, portfolio, paths):
    # E[D(0, T0) V(T0)] = V(0) for a swap with no cash flows before its start T0
    swap = trade(portfolio, "A3")
    cube = revalue_portfolio([swap], curve, paths)
    k = time_index(paths.times, swap.start)[0]
    discounted = paths.discount_factors(curve)[:, k] * cube[0, :, k]
    se = discounted.std(ddof=1) / np.sqrt(discounted.size)

    assert abs(discounted.mean() - swap_mtm(swap, curve)) <= Z_TOL * se


@pytest.mark.parametrize("t", [1.0, 2.0 + 1 / 12, 5.5])
def test_discounted_swap_value_is_a_martingale(curve, portfolio, paths, t):
    # E[D(0, t) V(t)] = time-0 PV of the cash flows still to come after t
    swap = trade(portfolio, "A1")
    cube = revalue_portfolio([swap], curve, paths)
    k = time_index(paths.times, t)[0]
    discounted = paths.discount_factors(curve)[:, k] * cube[0, :, k]
    se = discounted.std(ddof=1) / np.sqrt(discounted.size)

    assert abs(discounted.mean() - remaining_pv(swap, curve, t)) <= Z_TOL * se


def test_netting_set_mtm_sums_member_trades(curve, portfolio, grid):
    sim = simulate_paths(HW_PARAMS, grid, 300, np.random.default_rng(3))
    cube = revalue_portfolio(portfolio, curve, sim)
    netted = netting_set_mtm(cube, portfolio)

    assert set(netted) == {"NS_A", "NS_B"}
    np.testing.assert_allclose(netted["NS_A"] + netted["NS_B"], cube.sum(axis=0), rtol=1e-12)
    members = [i for i, swap in enumerate(portfolio) if swap.netting_set_id == "NS_B"]
    np.testing.assert_allclose(netted["NS_B"], cube[members].sum(axis=0), rtol=1e-12)


def test_flipping_directions_negates_the_cube(curve, portfolio, grid):
    sim = simulate_paths(HW_PARAMS, grid, 200, np.random.default_rng(5))
    flipped = [replace(s, direction="receiver" if s.direction == "payer" else "payer") for s in portfolio]

    np.testing.assert_allclose(revalue_portfolio(flipped, curve, sim), -revalue_portfolio(portfolio, curve, sim), atol=1e-6)


# ---------------------------------------------------------------- path refinement (Brownian bridge)

from src.rate_models import integral_variance, x_variance, zcb_price  # noqa: E402
from src.simulation import refine_paths  # noqa: E402


@pytest.fixture(scope="module")
def coarse_and_fine():
    coarse_grid = np.arange(0.0, 6.0 + 1e-9, 1.0)
    new_dates = np.array([0.3, 0.6, 2.5, 4.97])  # two points inside one interval, one near an end
    coarse = simulate_paths(HW_PARAMS, coarse_grid, 40_000, np.random.default_rng(31))
    fine = refine_paths(coarse, np.concatenate([coarse_grid, new_dates]), np.random.default_rng(32))
    return coarse, fine, new_dates


def test_refined_paths_keep_every_original_date(coarse_and_fine):
    coarse, fine, _ = coarse_and_fine
    idx = time_index(fine.times, coarse.times)

    np.testing.assert_array_equal(fine.x[:, idx], coarse.x)
    np.testing.assert_array_equal(fine.integral_x[:, idx], coarse.integral_x)


def test_bridged_dates_have_the_exact_marginal_law(coarse_and_fine):
    _, fine, new_dates = coarse_and_fine
    var_se = np.sqrt(2.0 / (fine.n_paths - 1))
    for s in new_dates:
        k = time_index(fine.times, s)[0]
        assert abs(fine.x[:, k].mean()) <= Z_TOL * np.sqrt(x_variance(HW_PARAMS, s) / fine.n_paths)
        assert abs(fine.x[:, k].var(ddof=1) / x_variance(HW_PARAMS, s) - 1) <= Z_TOL * var_se
        assert abs(fine.integral_x[:, k].var(ddof=1) / integral_variance(HW_PARAMS, s) - 1) <= Z_TOL * var_se


def test_bridged_dates_have_the_right_autocorrelation(coarse_and_fine):
    # Cov(x(s), x(t)) = e^{-a (t - s)} Var x(s) for s < t, across a bridged and an original date
    _, fine, _ = coarse_and_fine
    a = HW_PARAMS.mean_reversion
    s, t = 2.5, 3.0
    xs, xt = fine.x[:, time_index(fine.times, s)[0]], fine.x[:, time_index(fine.times, t)[0]]
    expected = np.exp(-a * (t - s)) * x_variance(HW_PARAMS, s)

    assert np.cov(xs, xt)[0, 1] == pytest.approx(expected, rel=Z_TOL * np.sqrt(2.0 / fine.n_paths))


def test_martingale_holds_at_bridged_dates(curve, coarse_and_fine):
    _, fine, _ = coarse_and_fine
    for s in (0.6, 4.97):
        k = time_index(fine.times, s)[0]
        discounted = fine.discount_factors(curve)[:, k] * zcb_price(curve, HW_PARAMS, s, np.array([6.0]), fine.x[:, k])[:, 0]
        se = discounted.std(ddof=1) / np.sqrt(discounted.size)
        assert abs(discounted.mean() - curve.df(6.0)) <= Z_TOL * se


def test_refine_rejects_grids_that_drop_or_extend_dates():
    coarse = simulate_paths(HW_PARAMS, np.array([0.0, 1.0, 2.0]), 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        refine_paths(coarse, np.array([0.0, 0.5, 2.0]), np.random.default_rng(1))  # drops 1.0
    with pytest.raises(ValueError):
        refine_paths(coarse, np.array([0.0, 1.0, 2.0, 2.5]), np.random.default_rng(1))  # beyond the last date


def test_refined_portfolio_values_match_on_original_dates(curve, portfolio, grid):
    from src.collateral import lookback_dates

    base = simulate_paths(HW_PARAMS, grid, 300, np.random.default_rng(9))
    fine_grid = np.unique(np.concatenate([grid, lookback_dates(grid, 14 / 365)]))
    fine = refine_paths(base, fine_grid, np.random.default_rng(10))
    fine_cube = revalue_portfolio(portfolio, curve, fine)

    np.testing.assert_array_equal(fine_cube[:, :, time_index(fine_grid, grid)], revalue_portfolio(portfolio, curve, base))
