from dataclasses import replace

import numpy as np
import pytest

from src.config import COUNTERPARTIES_FILE, CURVE_FILE, HW_PARAMS, PORTFOLIO_FILE
from src.credit import HazardCurve, load_counterparties
from src.curves import load_treasury_curve
from src.cva import CvaEstimate, aggregate_buckets, cva_by_bucket, discounted_ee, estimate_cva, pathwise_cva, summary_frame
from src.exposure import gross_exposure, netted_exposure
from src.instruments import load_portfolio
from src.rate_models import HullWhiteParams
from src.simulation import build_time_grid, revalue_portfolio, simulate_paths

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
    return simulate_paths(HW_PARAMS, grid, 4_000, np.random.default_rng(8))


@pytest.fixture(scope="module")
def netted(curve, portfolio, paths):
    return netted_exposure(revalue_portfolio(portfolio, curve, paths), portfolio)


@pytest.fixture(scope="module")
def credits():
    return load_counterparties(COUNTERPARTIES_FILE)


def cva_value(exposure, curve, grid, hazard, lgd):
    return estimate_cva(exposure, curve.df(grid), HazardCurve.flat(hazard).marginal_pd(grid), lgd).value


def test_zero_hazard_zero_cva(curve, grid, netted):
    for exposure in netted.values():
        assert cva_value(exposure, curve, grid, 0.0, 0.6) == 0.0


def test_full_recovery_zero_cva(curve, grid, netted):
    for exposure in netted.values():
        assert cva_value(exposure, curve, grid, 0.05, 0.0) == 0.0


def test_cva_monotone_in_hazard(curve, grid, netted):
    hazards = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    for exposure in netted.values():
        cvas = [cva_value(exposure, curve, grid, h, 0.6) for h in hazards]
        assert np.all(np.diff(cvas) > 0)


def test_cva_monotone_in_recovery(curve, grid, netted):
    # Hazard held fixed: only LGD changes, so CVA is exactly proportional to 1 - R
    recoveries = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    for exposure in netted.values():
        cvas = np.array([cva_value(exposure, curve, grid, 0.03, 1 - r) for r in recoveries])
        assert np.all(np.diff(cvas) < 0)
        np.testing.assert_allclose(cvas, cvas[0] * (1 - recoveries), rtol=1e-12)


def test_cva_monotone_in_notional(curve, portfolio, paths, grid, netted):
    # Doubling every notional doubles every value, exposure and CVA on the same paths
    doubled = [replace(swap, notional=2 * swap.notional) for swap in portfolio]
    netted_doubled = netted_exposure(revalue_portfolio(doubled, curve, paths), doubled)
    for ns in netted:
        base = cva_value(netted[ns], curve, grid, 0.03, 0.6)
        assert cva_value(netted_doubled[ns], curve, grid, 0.03, 0.6) == pytest.approx(2 * base, rel=1e-10)


def test_netted_cva_not_above_gross(curve, portfolio, paths, grid, netted, credits):
    gross = gross_exposure(revalue_portfolio(portfolio, curve, paths), portfolio)
    for ns, credit in credits.items():
        mpd = credit.hazard_curve().marginal_pd(grid)
        assert np.all(pathwise_cva(netted[ns], curve.df(grid), mpd, credit.lgd)
                      <= pathwise_cva(gross[ns], curve.df(grid), mpd, credit.lgd) + 1e-9)


def test_bucket_contributions_sum_to_total(curve, grid, netted, credits):
    for ns, credit in credits.items():
        mpd = credit.hazard_curve().marginal_pd(grid)
        buckets = cva_by_bucket(discounted_ee(netted[ns], curve.df(grid)), mpd, credit.lgd)
        total = estimate_cva(netted[ns], curve.df(grid), mpd, credit.lgd).value

        assert buckets.sum() == pytest.approx(total, rel=1e-12)
        yearly = aggregate_buckets(grid, buckets, np.arange(0.0, 11.0))
        assert yearly.sum() == pytest.approx(total, rel=1e-12)


def test_constant_exposure_cva_is_lgd_times_exposure_times_pd():
    grid = np.linspace(0.0, 5.0, 61)
    exposure = np.full((3, grid.size), 2.0e6)
    curve = HazardCurve.flat(0.04)
    cva = estimate_cva(exposure, np.ones(grid.size), curve.marginal_pd(grid), 0.6)

    assert cva.value == pytest.approx(0.6 * 2.0e6 * curve.cumulative_pd(5.0), rel=1e-12)
    assert cva.std_error <= 1e-12 * cva.value  # identical paths: zero variance up to float rounding


def test_discretized_cva_matches_continuous_closed_form():
    # Constant exposure E, flat rate r, flat hazard h:
    # CVA = LGD * E * int_0^T e^{-rt} h e^{-ht} dt = LGD * E * h / (r + h) * (1 - e^{-(r+h)T})
    r, h, lgd, e, horizon = 0.05, 0.03, 0.6, 1.0e6, 10.0
    grid = np.linspace(0.0, horizon, 1201)
    cva = estimate_cva(np.full((1, grid.size), e), np.exp(-r * grid), HazardCurve.flat(h).marginal_pd(grid), lgd)
    closed_form = lgd * e * h / (r + h) * (1 - np.exp(-(r + h) * horizon))

    assert cva.value == pytest.approx(closed_form, rel=1e-5)


def test_aggregate_buckets_assigns_intervals_by_end_date():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    by_interval = np.array([1.0, 2.0, 3.0, 4.0])  # intervals ending at 0.5, 1.0, 1.5, 2.0

    np.testing.assert_allclose(aggregate_buckets(times, by_interval, np.array([0.0, 1.0, 2.0])), [3.0, 7.0])


def test_summary_total_adds_netting_sets():
    rows = [
        {"netting_set_id": "X", "counterparty_id": "C1", "cva_netted": 10.0, "cva_netted_se": 1.0, "cva_gross": 20.0,
         "cva_benchmark_curve_df": 11.0, "integration_intervals": 120},
        {"netting_set_id": "Y", "counterparty_id": "C2", "cva_netted": 30.0, "cva_netted_se": 2.0, "cva_gross": 40.0,
         "cva_benchmark_curve_df": 33.0, "integration_intervals": 120},
    ]
    total = summary_frame(rows, CvaEstimate(40.0, 2.5, 100)).iloc[-1]

    assert total["netting_set_id"] == "TOTAL"
    assert total["cva_netted"] == 40.0 and total["cva_gross"] == 60.0
    assert total["netting_reduction"] == pytest.approx(1 / 3)
    assert total["benchmark_vs_headline"] == pytest.approx(44.0 / 40.0 - 1)


def test_pathwise_discounting_equals_curve_discounting_when_rates_are_deterministic(curve, portfolio, grid, credits):
    # sigma = 0: D(0,t) = P(0,t) on every path, so the headline and the benchmark coincide
    flat = HullWhiteParams(HW_PARAMS.mean_reversion, 0.0)
    sim = simulate_paths(flat, grid, 3, np.random.default_rng(0))
    netted = netted_exposure(revalue_portfolio(portfolio, curve, sim), portfolio)
    np.testing.assert_allclose(sim.discount_factors(curve), np.broadcast_to(curve.df(grid), (3, grid.size)), rtol=1e-12)
    for ns, credit in credits.items():
        mpd = credit.hazard_curve().marginal_pd(grid)
        pathwise = estimate_cva(netted[ns], sim.discount_factors(curve), mpd, credit.lgd).value
        assert pathwise == pytest.approx(estimate_cva(netted[ns], curve.df(grid), mpd, credit.lgd).value, rel=1e-12)


def test_cva_integrates_over_every_simulation_interval(curve, paths, grid, netted, credits):
    # The integral runs over the full simulation grid; yearly buckets only re-group it.
    credit = credits["NS_B"]
    mpd = credit.hazard_curve().marginal_pd(grid)
    by_interval = cva_by_bucket(discounted_ee(netted["NS_B"], paths.discount_factors(curve)), mpd, credit.lgd)

    assert mpd.size == by_interval.size == grid.size - 1
    yearly = aggregate_buckets(grid, by_interval, np.arange(0.0, 11.0))
    annual_grid = np.arange(0.0, 11.0)  # integrating on the coarse grid instead would give a different number
    coarse = estimate_cva(netted["NS_B"][:, np.searchsorted(grid, annual_grid)],
                          paths.discount_factors(curve)[:, np.searchsorted(grid, annual_grid)],
                          credit.hazard_curve().marginal_pd(annual_grid), credit.lgd).value
    assert yearly.sum() == pytest.approx(by_interval.sum(), rel=1e-12)
    assert abs(coarse / by_interval.sum() - 1) > 1e-3


@pytest.mark.slow
def test_standard_error_shrinks(curve, portfolio, grid, credits):
    # Monte Carlo error should fall like 1/sqrt(N): 4x the paths -> about half the standard error
    credit = credits["NS_B"]
    mpd = credit.hazard_curve().marginal_pd(grid)
    estimates = []
    for n, seed in [(1_000, 101), (4_000, 202), (16_000, 303)]:
        sim = simulate_paths(HW_PARAMS, grid, n, np.random.default_rng(seed))
        exposure = netted_exposure(revalue_portfolio(portfolio, curve, sim), portfolio)["NS_B"]
        estimates.append(estimate_cva(exposure, curve.df(grid), mpd, credit.lgd))

    for small, large in zip(estimates, estimates[1:]):
        assert 0.4 < large.std_error / small.std_error < 0.6
        assert abs(large.value - small.value) <= Z_TOL * np.hypot(large.std_error, small.std_error)
