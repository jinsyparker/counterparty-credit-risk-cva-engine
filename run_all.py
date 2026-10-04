"""Reproducible runner: regenerates every figure and table in ``reports/``.

Usage (from the project root):
    python run_all.py           # base case (10,000 paths)
    python run_all.py --quick   # 2,000 paths, for fast iteration

Stages are added phase by phase; each one reads its inputs from ``data/`` and
``src/config.py`` and writes only to ``reports/``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # headless: figures are only saved, never shown

import numpy as np
import pandas as pd

from src.config import (
    BASE_SIMULATION,
    BRIDGE_SEED_OFFSET,
    CHUNK_SEED_STRIDE,
    CONVERGENCE_PATHS,
    COUNTERPARTIES_FILE,
    CSA_SCENARIOS,
    CURVE_FILE,
    DISTRIBUTION_TIMES,
    FIGURES_DIR,
    HW_PARAMS,
    MATURITY_SCALES,
    MEAN_REVERSION_GRID,
    MPOR_GRID_DAYS,
    NOTIONAL_SCALES,
    PFE_QUANTILE,
    PORTFOLIO_FILE,
    PROJECT_ROOT,
    QUICK_SIMULATION,
    RECOVERY_GRID,
    REPORTS_DIR,
    SIGMA_GRID,
    SPREAD_GRID_BP,
    TABLES_DIR,
    VALUATION_DATE,
)
from src.collateral import collateral_analysis
from src.credit import credit_curve_table, load_counterparties
from src.curves import DiscountCurve, curve_table, load_par_yields, load_treasury_curve
from src.cva import (
    CvaEstimate,
    aggregate_buckets,
    cva_by_bucket,
    cva_summary_row,
    discounted_ee,
    estimate_cva,
    pathwise_cva,
    summary_frame,
)
from src.exposure import exposure_summary, gross_exposure, netted_exposure, netting_set_horizons, profile_table
from src.instruments import Swap, load_portfolio
from src.plotting import (
    plot_collateral_profiles,
    plot_convergence,
    plot_credit_curves,
    plot_credit_sensitivity,
    plot_curve,
    plot_cva_contributions,
    plot_cva_mitigation,
    plot_exposure_distributions,
    plot_exposure_profiles,
    plot_gross_vs_netted,
    plot_market_sensitivity,
    plot_mtm_fan,
    plot_rate_paths,
    plot_sensitivity_tornado,
    save_figure,
)
from src.rate_models import phi_adjustment
from src.simulation import RatePaths, SimulationSettings, build_time_grid, netting_set_mtm, revalue_portfolio, simulate_paths
from src.swap_pricer import cashflow_table, netting_set_summary, par_rate, price_portfolio, swap_mtm
from src.reporting import update_readme_results, write_final_summary
from src.validation import (
    convergence_table,
    credit_sensitivity_table,
    market_sensitivity_table,
    martingale_table,
    percentile_profile,
    rate_distribution_table,
    tornado_table,
)

EXAMPLE_TRADE = "A1"


def stage_a_pricing() -> tuple[DiscountCurve, list[Swap]]:
    """Phase A: curve bootstrap, example swap cash flows, portfolio valuation."""
    par_yields = load_par_yields(CURVE_FILE)
    curve = load_treasury_curve(CURVE_FILE)
    portfolio = load_portfolio(PORTFOLIO_FILE)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    curve_table(curve).to_csv(TABLES_DIR / "zero_curve.csv", index=False)
    save_figure(plot_curve(curve, par_yields), FIGURES_DIR / "yield_curve.png")

    example = next(swap for swap in portfolio if swap.trade_id == EXAMPLE_TRADE)
    cashflow_table(example, curve).to_csv(TABLES_DIR / "swap_cashflows.csv", index=False)

    valuation = price_portfolio(portfolio, curve)
    valuation.to_csv(TABLES_DIR / "portfolio_valuation.csv", index=False)
    summary = netting_set_summary(valuation)
    summary.to_csv(TABLES_DIR / "netting_set_mtm.csv", index=False)

    # Sanity checks printed with every run (the real assertions live in tests/)
    at_par = replace(example, fixed_rate=par_rate(example, curve))
    flipped = replace(example, direction="receiver" if example.direction == "payer" else "payer")
    print(f"[A] Valuation date {VALUATION_DATE}: {len(curve.times)} curve nodes, {len(portfolio)} trades")
    print(f"    {EXAMPLE_TRADE} at par rate {at_par.fixed_rate:.4%}: MtM = {swap_mtm(at_par, curve):,.6f}")
    print(f"    {EXAMPLE_TRADE} payer vs receiver: {swap_mtm(example, curve):,.2f} / {swap_mtm(flipped, curve):,.2f}")
    with pd.option_context("display.width", 120, "display.float_format", "{:,.2f}".format):
        print(summary.to_string(index=False))
    return curve, portfolio


def stage_b_simulation(curve: DiscountCurve, portfolio: list[Swap], settings: SimulationSettings) -> tuple[RatePaths, np.ndarray]:
    """Phase B: Hull-White paths, distribution and martingale checks, MtM cube."""
    start = time.perf_counter()
    grid = build_time_grid(portfolio, step=settings.time_step)
    paths = simulate_paths(HW_PARAMS, grid, settings.n_paths, np.random.default_rng(settings.seed))
    cube = revalue_portfolio(portfolio, curve, paths)
    elapsed = time.perf_counter() - start

    short_rates = paths.short_rates(curve)
    analytic_mean = curve.inst_forward(grid) + phi_adjustment(HW_PARAMS, grid)
    save_figure(plot_rate_paths(grid, short_rates, analytic_mean), FIGURES_DIR / "rate_paths.png")
    mtm_by_set = netting_set_mtm(cube, portfolio)
    save_figure(plot_mtm_fan(grid, mtm_by_set), FIGURES_DIR / "mtm_paths.png")

    martingale = martingale_table(curve, paths)
    rate_distribution_table(curve, paths).to_csv(TABLES_DIR / "rate_distribution.csv", index=False)
    martingale.to_csv(TABLES_DIR / "martingale_check.csv", index=False)
    percentile_profile(grid, mtm_by_set).to_csv(TABLES_DIR / "mtm_summary.csv", index=False)

    t0_error = np.max(np.abs(cube[:, :, 0] - np.array([swap_mtm(s, curve) for s in portfolio])[:, None]))
    print(f"[B] Hull-White a={HW_PARAMS.mean_reversion}, sigma={HW_PARAMS.volatility}: {settings.n_paths:,} paths x "
          f"{grid.size} dates, simulated + revalued in {elapsed:.1f}s")
    print(f"    cube at t=0 vs time-0 pricer: max abs difference {t0_error:.2e}")
    print(f"    martingale z-scores: {np.round(martingale['z_score'].to_numpy(), 2).tolist()}")
    return paths, cube


def stage_c_exposure(portfolio: list[Swap], paths: RatePaths, cube: np.ndarray) -> None:
    """Phase C: netted and gross exposure, EE / PFE / EPE profiles, exposure distributions."""
    grid = paths.times
    netted, gross = netted_exposure(cube, portfolio), gross_exposure(cube, portfolio)
    profiles = profile_table(grid, netted, gross, PFE_QUANTILE)
    summary = exposure_summary(grid, netted, gross, netting_set_horizons(portfolio), PFE_QUANTILE)
    profiles.to_csv(TABLES_DIR / "exposure_profiles.csv", index=False)
    summary.to_csv(TABLES_DIR / "exposure_summary.csv", index=False)

    by_set = {ns: rows for ns, rows in profiles.groupby("netting_set_id", sort=False)}
    ee = {ns: rows["ee_netted"].to_numpy() for ns, rows in by_set.items()}
    save_figure(plot_exposure_profiles(grid, ee, {ns: rows["pfe_netted"].to_numpy() for ns, rows in by_set.items()},
                                       PFE_QUANTILE), FIGURES_DIR / "exposure_profiles.png")
    save_figure(plot_gross_vs_netted(grid, ee, {ns: rows["ee_gross"].to_numpy() for ns, rows in by_set.items()}),
                FIGURES_DIR / "gross_vs_netted_ee.png")
    save_figure(plot_exposure_distributions(grid, netting_set_mtm(cube, portfolio), DISTRIBUTION_TIMES, PFE_QUANTILE),
                FIGURES_DIR / "exposure_distributions.png")

    print(f"[C] Exposure (PFE at {PFE_QUANTILE:.0%}):")
    for row in summary.itertuples():
        print(f"    {row.netting_set_id}: EPE (time-average EE) netted {row.epe_netted:,.0f} vs gross {row.epe_gross:,.0f} "
              f"({row.netting_reduction:.0%} netting reduction); peak EE {row.peak_ee_netted:,.0f} at "
              f"{row.peak_ee_time:.2f}Y; peak PFE {getattr(row, f'peak_pfe{PFE_QUANTILE * 100:g}_netted'):,.0f} "
              f"at {row.peak_pfe_time:.2f}Y")


def stage_d_cva(curve: DiscountCurve, portfolio: list[Swap], paths: RatePaths, cube: np.ndarray) -> None:
    """Phase D: survival curves, unilateral CVA (netted vs gross), bucket contributions, credit sensitivities.

    Headline CVA discounts pathwise with the simulated bank account D(0, t); the
    deterministic P(0, t) x EE version is kept as a benchmark. CVA is integrated over
    every interval of the simulation grid and only aggregated to years for display.
    """
    grid = paths.times
    credits = load_counterparties(COUNTERPARTIES_FILE)
    horizons = netting_set_horizons(portfolio)
    netted, gross = netted_exposure(cube, portfolio), gross_exposure(cube, portfolio)
    curve_df, bank_df = curve.df(grid), paths.discount_factors(curve)
    edges = np.arange(0.0, np.ceil(grid[-1]) + 1.0)

    rows, buckets, total_per_path = [], [], np.zeros(paths.n_paths)
    yearly_netted, yearly_gross = {}, {}
    for ns, credit in credits.items():
        mpd = credit.hazard_curve().marginal_pd(grid)
        cva_net = estimate_cva(netted[ns], bank_df, mpd, credit.lgd)
        cva_gross = estimate_cva(gross[ns], bank_df, mpd, credit.lgd)
        cva_benchmark = estimate_cva(netted[ns], curve_df, mpd, credit.lgd)
        total_per_path += pathwise_cva(netted[ns], bank_df, mpd, credit.lgd)
        rows.append(cva_summary_row(ns, credit.counterparty_id, cva_net, cva_gross, cva_benchmark, credit.spread,
                                    credit.recovery, credit.hazard, float(credit.hazard_curve().cumulative_pd(horizons[ns])),
                                    n_intervals=mpd.size))
        yearly_netted[ns] = aggregate_buckets(grid, cva_by_bucket(discounted_ee(netted[ns], bank_df), mpd, credit.lgd), edges)
        yearly_gross[ns] = aggregate_buckets(grid, cva_by_bucket(discounted_ee(gross[ns], bank_df), mpd, credit.lgd), edges)
        buckets.append(pd.DataFrame({"netting_set_id": ns, "bucket_start": edges[:-1], "bucket_end": edges[1:],
                                     "cva_netted": yearly_netted[ns], "cva_gross": yearly_gross[ns]}))

    n = total_per_path.size
    total = CvaEstimate(float(total_per_path.mean()), float(total_per_path.std(ddof=1) / np.sqrt(n)), n)
    summary = summary_frame(rows, total)
    summary.to_csv(TABLES_DIR / "cva_summary.csv", index=False)
    pd.concat(buckets, ignore_index=True).to_csv(TABLES_DIR / "cva_buckets.csv", index=False)
    credit_curve_table(credits, edges).to_csv(TABLES_DIR / "credit_curves.csv", index=False)
    sensitivity = credit_sensitivity_table(grid, netted, bank_df, credits, SPREAD_GRID_BP, RECOVERY_GRID)
    sensitivity.to_csv(TABLES_DIR / "cva_credit_sensitivity.csv", index=False)

    save_figure(plot_credit_curves(credits, horizon=edges[-1]), FIGURES_DIR / "credit_curves.png")
    save_figure(plot_cva_contributions(edges, yearly_netted, yearly_gross), FIGURES_DIR / "cva_contributions.png")
    save_figure(plot_credit_sensitivity(sensitivity), FIGURES_DIR / "cva_credit_sensitivity.png")

    print(f"[D] Unilateral CVA, pathwise discounting, integrated over {grid.size - 1} grid intervals (± one std error):")
    for row in summary.itertuples():
        print(f"    {row.netting_set_id:5s} netted {row.cva_netted:>10,.0f} ± {row.cva_netted_se:,.0f}   gross "
              f"{row.cva_gross:>10,.0f}   netting -{row.netting_reduction:.0%}   curve-DF benchmark "
              f"{row.cva_benchmark_curve_df:>10,.0f} ({row.benchmark_vs_headline:+.1%})")


def stage_e_collateral(curve: DiscountCurve, portfolio: list[Swap], paths: RatePaths, settings: SimulationSettings) -> None:
    """Phase E: one-way CSA scenarios, collateralized EE and CVA, gross -> netted -> collateralized, MPoR sensitivity."""
    start = time.perf_counter()
    result = collateral_analysis(curve, portfolio, paths, load_counterparties(COUNTERPARTIES_FILE), CSA_SCENARIOS,
                                 MPOR_GRID_DAYS, np.random.default_rng(settings.seed + BRIDGE_SEED_OFFSET), PFE_QUANTILE)
    elapsed = time.perf_counter() - start
    result["summary"].to_csv(TABLES_DIR / "collateral_summary.csv", index=False)
    result["mitigation"].to_csv(TABLES_DIR / "cva_mitigation.csv", index=False)
    result["mpor"].to_csv(TABLES_DIR / "collateral_mpor_sensitivity.csv", index=False)
    result["profiles"].to_csv(TABLES_DIR / "collateral_profiles.csv", index=False)
    save_figure(plot_collateral_profiles(result["profiles"]), FIGURES_DIR / "collateral_ee_profiles.png")
    save_figure(plot_cva_mitigation(result["mitigation"]), FIGURES_DIR / "cva_mitigation.png")

    print(f"[E] Collateral: {len(CSA_SCENARIOS)} CSA scenarios on the Phase D paths ({result['fine_grid_size']} dates with "
          f"look-backs, base dates unchanged: {result['base_dates_unchanged']}), {elapsed:.1f}s")
    for row in result["mitigation"].itertuples():
        print(f"    {row.netting_set_id}  {row.stage:45s} CVA {row.cva:>10,.0f} ± {row.cva_se:>6,.0f}   "
              f"-{row.reduction_vs_gross:.0%} vs gross")


def stage_f_validation(curve: DiscountCurve, portfolio: list[Swap], paths: RatePaths, cube: np.ndarray,
                       settings: SimulationSettings, write_documents: bool) -> None:
    """Phase F: Monte Carlo convergence, market sensitivities, tornado, final summary and README results."""
    start = time.perf_counter()
    credits = load_counterparties(COUNTERPARTIES_FILE)
    counts = tuple(n for n in CONVERGENCE_PATHS if n <= 5 * settings.n_paths)
    convergence = convergence_table(curve, portfolio, credits, HW_PARAMS, settings, counts, CHUNK_SEED_STRIDE)
    market = market_sensitivity_table(curve, portfolio, credits, HW_PARAMS, settings, SIGMA_GRID, MEAN_REVERSION_GRID,
                                      MATURITY_SCALES, NOTIONAL_SCALES)
    tornado = tornado_table(market, paths.times, netted_exposure(cube, portfolio), paths.discount_factors(curve), credits)
    convergence.to_csv(TABLES_DIR / "convergence.csv", index=False)
    market.to_csv(TABLES_DIR / "market_sensitivity.csv", index=False)
    tornado.to_csv(TABLES_DIR / "sensitivity_tornado.csv", index=False)
    save_figure(plot_convergence(convergence), FIGURES_DIR / "convergence.png")
    save_figure(plot_market_sensitivity(market), FIGURES_DIR / "market_sensitivity.png")
    save_figure(plot_sensitivity_tornado(tornado), FIGURES_DIR / "sensitivity_tornado.png")
    elapsed = time.perf_counter() - start

    total = convergence[convergence["netting_set_id"] == "TOTAL"].sort_values("n_paths")
    print(f"[F] Validation ({elapsed:.1f}s): portfolio CVA {total['cva'].iloc[-1]:,.0f} ± {total['cva_se'].iloc[-1]:,.0f} "
          f"at {total['n_paths'].iloc[-1]:,} paths (base {settings.n_paths:,}: {total.loc[total.n_paths == settings.n_paths, 'cva'].iloc[0]:,.0f})")
    for parameter, rows in tornado.groupby("parameter", sort=False):
        print(f"    {parameter:22s} " + "   ".join(f"{v:g}: {c:+.0%}" for v, c in zip(rows["value"], rows["change_vs_base"])))
    if write_documents:
        n_dates = paths.times.size
        write_final_summary(REPORTS_DIR / "final_project_summary.md", TABLES_DIR, settings.n_paths, n_dates, VALUATION_DATE)
        updated = update_readme_results(PROJECT_ROOT / "README.md", TABLES_DIR, settings.n_paths, n_dates, VALUATION_DATE)
        print(f"    wrote reports/final_project_summary.md; README results block {'updated' if updated else 'markers not found'}")
    else:
        print("    --quick: final summary and README results left unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="use fewer paths for fast iteration")
    args = parser.parse_args()
    settings = QUICK_SIMULATION if args.quick else BASE_SIMULATION

    curve, portfolio = stage_a_pricing()
    paths, cube = stage_b_simulation(curve, portfolio, settings)
    stage_c_exposure(portfolio, paths, cube)
    stage_d_cva(curve, portfolio, paths, cube)
    stage_e_collateral(curve, portfolio, paths, settings)
    stage_f_validation(curve, portfolio, paths, cube, settings, write_documents=not args.quick)
    print(f"Outputs written to {FIGURES_DIR.parent}")


if __name__ == "__main__":
    main()
