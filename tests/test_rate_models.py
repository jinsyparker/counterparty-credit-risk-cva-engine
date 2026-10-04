import numpy as np
import pytest

from src.config import CURVE_FILE, HW_PARAMS
from src.curves import load_treasury_curve
from src.rate_models import HullWhiteParams, integral_variance, x_variance, zcb_price
from src.simulation import simulate_paths, time_index

N_PATHS = 20_000
Z_TOL = 4.0  # Monte Carlo tolerance in standard errors (fixed seed)
# Non-uniform grid on purpose: exact simulation must not depend on step sizes
GRID = np.unique(np.round(np.concatenate([np.arange(0, 10 + 1e-9, 1 / 12), [0.1, 0.35, 2.9, 7.3]]), 10))


@pytest.fixture(scope="module")
def curve():
    return load_treasury_curve(CURVE_FILE)


@pytest.fixture(scope="module")
def paths():
    return simulate_paths(HW_PARAMS, GRID, N_PATHS, np.random.default_rng(2024))


def assert_mc_close(samples, expected):
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    assert abs(samples.mean() - expected) <= Z_TOL * se, f"mean {samples.mean()} vs {expected} (se {se})"


@pytest.mark.parametrize("a, sigma", [(0.0, 0.01), (-0.1, 0.01), (0.03, -0.01)])
def test_invalid_params_raise(a, sigma):
    with pytest.raises(ValueError):
        HullWhiteParams(a, sigma)


def test_time_zero_bond_prices_reproduce_curve(curve):
    maturities = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    prices = zcb_price(curve, HW_PARAMS, 0.0, maturities, np.zeros(3))

    np.testing.assert_allclose(prices, np.broadcast_to(curve.df(maturities), (3, 5)), rtol=1e-14)


def test_bond_maturing_now_is_worth_one(curve):
    prices = zcb_price(curve, HW_PARAMS, 3.0, np.array([3.0]), np.array([-0.02, 0.0, 0.03]))

    np.testing.assert_allclose(prices, 1.0, rtol=1e-14)


def test_zero_volatility_bond_price_is_forward_discount_factor(curve):
    flat = HullWhiteParams(0.03, 0.0)
    prices = zcb_price(curve, flat, 2.0, np.array([3.0, 7.0]), np.zeros(1))

    np.testing.assert_allclose(prices[0], curve.df(np.array([3.0, 7.0])) / curve.df(2.0), rtol=1e-14)


def test_higher_x_means_lower_bond_prices(curve):
    prices = zcb_price(curve, HW_PARAMS, 1.0, np.array([5.0]), np.array([-0.01, 0.0, 0.01]))[:, 0]

    assert np.all(np.diff(prices) < 0)


@pytest.mark.parametrize("t", [0.35, 1.0, 2.9, 5.0, 10.0])
def test_simulated_moments_match_analytic(paths, t):
    k = time_index(GRID, t)[0]
    x, integral = paths.x[:, k], paths.integral_x[:, k]
    a, sigma = HW_PARAMS.mean_reversion, HW_PARAMS.volatility
    var_x, var_i = x_variance(HW_PARAMS, t), integral_variance(HW_PARAMS, t)
    cov_xi = sigma**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2
    var_se = np.sqrt(2.0 / (N_PATHS - 1))  # relative standard error of a sample variance

    assert_mc_close(x, 0.0)
    assert_mc_close(integral, 0.0)
    assert abs(x.var(ddof=1) / var_x - 1) <= Z_TOL * var_se
    assert abs(integral.var(ddof=1) / var_i - 1) <= Z_TOL * var_se
    assert np.cov(x, integral)[0, 1] == pytest.approx(cov_xi, rel=Z_TOL * var_se)


def test_mean_reversion_speed_is_recovered_by_regression():
    # E[x(t+h) | x(t)] = x(t) e^{-a h}: regress x(t+h) on x(t) across paths
    strong = HullWhiteParams(0.5, 0.01)
    grid = np.array([0.0, 1.0, 3.0])
    sim = simulate_paths(strong, grid, N_PATHS, np.random.default_rng(7))
    x_now, x_later = sim.x[:, 1], sim.x[:, 2]
    slope = np.dot(x_now, x_later) / np.dot(x_now, x_now)
    residual_sd = np.std(x_later - slope * x_now)
    slope_se = residual_sd / np.sqrt(np.dot(x_now, x_now))

    assert abs(slope - np.exp(-0.5 * 2.0)) <= Z_TOL * slope_se


@pytest.mark.parametrize("t", [1.0, 5.0, 10.0])
def test_expected_stochastic_discount_equals_curve(curve, paths, t):
    k = time_index(GRID, t)[0]

    assert_mc_close(paths.discount_factors(curve)[:, k], curve.df(t))


@pytest.mark.parametrize("t, maturity", [(1.0, 5.0), (2.9, 7.3), (5.0, 10.0), (9.5, 10.0)])
def test_discounted_bond_price_is_a_martingale(curve, paths, t, maturity):
    # E[D(0,t) P(t,T)] = P(0,T): the model is arbitrage-free and fits today's curve
    k = time_index(GRID, t)[0]
    discounted = paths.discount_factors(curve)[:, k] * zcb_price(curve, HW_PARAMS, t, np.array([maturity]), paths.x[:, k])[:, 0]

    assert_mc_close(discounted, curve.df(maturity))


def test_same_seed_reproduces_paths():
    first = simulate_paths(HW_PARAMS, GRID, 100, np.random.default_rng(42))
    second = simulate_paths(HW_PARAMS, GRID, 100, np.random.default_rng(42))

    np.testing.assert_array_equal(first.x, second.x)


def test_common_random_numbers_scale_paths_with_volatility():
    # Same seed and grid, so the same shocks: x is linear in sigma
    low = simulate_paths(HullWhiteParams(0.03, 0.01), GRID, 100, np.random.default_rng(42))
    high = simulate_paths(HullWhiteParams(0.03, 0.02), GRID, 100, np.random.default_rng(42))

    np.testing.assert_allclose(high.x, 2.0 * low.x, rtol=1e-12, atol=1e-15)
