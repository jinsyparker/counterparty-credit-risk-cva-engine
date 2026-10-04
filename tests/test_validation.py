import numpy as np
import pytest

from src.config import COUNTERPARTIES_FILE, CURVE_FILE, HW_PARAMS, PORTFOLIO_FILE, TABLES_DIR
from src.credit import load_counterparties
from src.curves import load_treasury_curve
from src.exposure import netted_exposure
from src.instruments import load_portfolio
from src.simulation import SimulationSettings, build_time_grid, revalue_portfolio, simulate_paths
from src.swap_pricer import par_rate
from src.validation import (
    convergence_table,
    headline_per_path,
    market_sensitivity_table,
    scale_maturities,
    tornado_table,
)

SMALL = SimulationSettings(n_paths=1_000, seed=5, time_step=1 / 12)


@pytest.fixture(scope="module")
def curve():
    return load_treasury_curve(CURVE_FILE)


@pytest.fixture(scope="module")
def portfolio():
    return load_portfolio(PORTFOLIO_FILE)


@pytest.fixture(scope="module")
def credits():
    return load_counterparties(COUNTERPARTIES_FILE)


@pytest.fixture(scope="module")
def market(curve, portfolio, credits):
    return market_sensitivity_table(curve, portfolio, credits, HW_PARAMS, SMALL, sigmas=(0.005, 0.01, 0.015),
                                    mean_reversions=(0.01, 0.03, 0.10), maturity_scales=(0.75, 1.0, 1.25),
                                    notional_scales=(0.5, 1.0, 1.5))


def cva_of(market, parameter, value, ns):
    rows = market[(market["parameter"] == parameter) & np.isclose(market["value"], value) & (market["netting_set_id"] == ns)]
    return float(rows["cva"].iloc[0])


def test_scaling_maturities_by_one_returns_the_same_trades(curve, portfolio):
    assert scale_maturities(portfolio, curve, 1.0) == portfolio


def test_scaled_trades_keep_their_offset_to_par(curve, portfolio):
    scaled = scale_maturities(portfolio, curve, 1.25)
    for old, new in zip(portfolio, scaled):
        assert new.start == old.start
        assert new.maturity - new.start == pytest.approx(round((old.maturity - old.start) * 1.25 * 4) / 4)
        assert new.fixed_rate - par_rate(new, curve) == pytest.approx(old.fixed_rate - par_rate(old, curve), abs=1e-12)


def test_first_convergence_chunk_is_the_base_run(curve, portfolio, credits):
    table = convergence_table(curve, portfolio, credits, HW_PARAMS, SMALL, (500, 1_000))
    direct = headline_per_path(curve, portfolio, credits, HW_PARAMS, SMALL.n_paths, SMALL.seed)
    row = table[(table["netting_set_id"] == "NS_B") & (table["n_paths"] == 1_000)].iloc[0]

    assert row["cva"] == pytest.approx(direct["NS_B"]["cva"].mean(), rel=1e-12)
    assert row["ci95_low"] < row["cva"] < row["ci95_high"]


def test_convergence_error_falls_like_one_over_root_n(curve, portfolio, credits):
    table = convergence_table(curve, portfolio, credits, HW_PARAMS, SimulationSettings(1_000, 11), (500, 2_000, 4_000))
    se = table[table["netting_set_id"] == "TOTAL"].set_index("n_paths")["cva_se"]

    assert 0.35 < se[2_000] / se[500] < 0.65


def test_headline_per_path_matches_stage_d_definition(curve, portfolio, credits):
    # Mean of per-path CVA = the pathwise-discounted CVA computed directly
    from src.cva import estimate_cva

    per_path = headline_per_path(curve, portfolio, credits, HW_PARAMS, 400, 3)
    grid = build_time_grid(portfolio)
    paths = simulate_paths(HW_PARAMS, grid, 400, np.random.default_rng(3))
    netted = netted_exposure(revalue_portfolio(portfolio, curve, paths), portfolio)
    credit = credits["NS_A"]
    direct = estimate_cva(netted["NS_A"], paths.discount_factors(curve), credit.hazard_curve().marginal_pd(grid), credit.lgd)

    assert per_path["NS_A"]["cva"].mean() == pytest.approx(direct.value, rel=1e-12)


def test_cva_rises_with_rate_volatility(market):
    for ns in ("NS_A", "NS_B", "TOTAL"):
        assert cva_of(market, "sigma", 0.005, ns) < cva_of(market, "sigma", 0.01, ns) < cva_of(market, "sigma", 0.015, ns)


def test_cva_falls_with_mean_reversion(market):
    for ns in ("NS_B", "TOTAL"):
        assert cva_of(market, "mean_reversion", 0.01, ns) > cva_of(market, "mean_reversion", 0.03, ns) > cva_of(market, "mean_reversion", 0.10, ns)


def test_cva_rises_with_portfolio_tenor(market):
    for ns in ("NS_A", "NS_B", "TOTAL"):
        assert cva_of(market, "maturity_scale", 0.75, ns) < cva_of(market, "maturity_scale", 1.0, ns) < cva_of(market, "maturity_scale", 1.25, ns)


def test_cva_is_linear_in_notional(market):
    for ns in ("NS_A", "NS_B", "TOTAL"):
        base = cva_of(market, "notional_scale", 1.0, ns)
        assert cva_of(market, "notional_scale", 1.5, ns) == pytest.approx(1.5 * base, rel=1e-12)


def test_tornado_credit_rows(curve, portfolio, credits, market):
    grid = build_time_grid(portfolio)
    paths = simulate_paths(HW_PARAMS, grid, SMALL.n_paths, np.random.default_rng(SMALL.seed))
    netted = netted_exposure(revalue_portfolio(portfolio, curve, paths), portfolio)
    table = tornado_table(market, grid, netted, paths.discount_factors(curve), credits,
                          market_ranges={"sigma": (0.005, 0.015), "maturity_scale": (0.75, 1.25)}).set_index(["parameter", "value"])

    # hazard fixed: CVA scales exactly with LGD (base recovery 40% for both counterparties)
    assert table.loc[("recovery_hazard_fixed", 0.2), "change_vs_base"] == pytest.approx(0.8 / 0.6 - 1, rel=1e-9)
    assert table.loc[("spread_scale", 0.5), "change_vs_base"] < 0 < table.loc[("spread_scale", 1.5), "change_vs_base"]


def test_reporting_reads_generated_tables():
    from src.reporting import load_results, resume_bullets, results_markdown

    if not (TABLES_DIR / "convergence.csv").exists():
        pytest.skip("run `python run_all.py` first to generate reports/tables")
    results = load_results(TABLES_DIR)
    block = results_markdown(results, 10_000, 121, "2026-10-02")

    assert "| Netting set |" in block and "**Portfolio**" in block
    assert len(resume_bullets(results, 10_000, 121)) == 2
