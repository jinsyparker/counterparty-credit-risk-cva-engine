"""Model checks and summaries shared by ``run_all.py`` and the notebooks.

* Phase B: simulated short-rate distribution vs the analytic Hull-White moments, and
  the martingale check E[D(0,t) P(t,T)] = P(0,T).
* Phase D: credit-only CVA sensitivities with the simulated exposures held fixed.
* Phase F: Monte Carlo convergence of CVA and EPE, and market sensitivities (sigma,
  mean reversion, portfolio maturity, notional) under common random numbers.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.curves import DiscountCurve
from src.credit import HazardCurve, hazard_from_spread
from src.cva import estimate_cva, pathwise_cva
from src.exposure import netted_exposure, netting_set_horizons
from src.instruments import Swap
from src.rate_models import HullWhiteParams, phi_adjustment, x_variance, zcb_price
from src.simulation import RatePaths, SimulationSettings, build_time_grid, revalue_portfolio, simulate_paths, time_index
from src.swap_pricer import par_rate

REPORT_TIMES = (0.5, 1.0, 2.0, 5.0, 10.0)
MARTINGALE_PAIRS = ((1.0, 2.0), (1.0, 5.0), (2.0, 10.0), (5.0, 10.0), (10.0, 10.0))


def rate_distribution_table(curve: DiscountCurve, paths: RatePaths, times: tuple[float, ...] = REPORT_TIMES) -> pd.DataFrame:
    """Simulated vs analytic mean / standard deviation of r(t), plus percentiles and P(r < 0).

    Analytic: E[r(t)] = phi(t) = f(0,t) + sigma^2/(2a^2)(1 - e^{-at})^2, Std[r(t)] = sqrt(Var[x(t)]).
    """
    short_rates = paths.short_rates(curve)
    rows = []
    for t in times:
        r = short_rates[:, time_index(paths.times, t)[0]]
        rows.append(
            {
                "time": t,
                "mean_mc": r.mean(),
                "mean_analytic": curve.inst_forward(t) + phi_adjustment(paths.params, t),
                "std_mc": r.std(ddof=1),
                "std_analytic": float(np.sqrt(x_variance(paths.params, t))),
                "p05": np.percentile(r, 5),
                "p95": np.percentile(r, 95),
                "prob_negative": (r < 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def martingale_table(curve: DiscountCurve, paths: RatePaths, pairs: tuple[tuple[float, float], ...] = MARTINGALE_PAIRS) -> pd.DataFrame:
    """E[D(0,t) P(t,T)] vs P(0,T) with Monte Carlo standard errors and z-scores.

    Holds exactly in an arbitrage-free model fitted to the curve, so a |z| well above
    3 would point to a bug in the bond formula, the discounting or the simulation.
    """
    discount = paths.discount_factors(curve)
    rows = []
    for t, maturity in pairs:
        k = time_index(paths.times, t)[0]
        sample = discount[:, k] * zcb_price(curve, paths.params, t, np.array([maturity]), paths.x[:, k])[:, 0]
        se = sample.std(ddof=1) / np.sqrt(sample.size)
        rows.append(
            {
                "t": t,
                "T": maturity,
                "mc_E[D(0,t)P(t,T)]": sample.mean(),
                "curve_P(0,T)": curve.df(maturity),
                "std_error": se,
                "z_score": (sample.mean() - curve.df(maturity)) / se,
            }
        )
    return pd.DataFrame(rows)


def percentile_profile(times: np.ndarray, values_by_key: dict[str, np.ndarray], report_times: tuple[float, ...] = REPORT_TIMES,
                       key_name: str = "netting_set_id") -> pd.DataFrame:
    """Mean and 5/95% percentiles of pathwise values (n_paths, n_times) at the report horizons."""
    rows = []
    for key, values in values_by_key.items():
        for t in report_times:
            column = values[:, time_index(times, t)[0]]
            rows.append({key_name: key, "time": t, "mean": column.mean(),
                         "p05": np.percentile(column, 5), "p95": np.percentile(column, 95)})
    return pd.DataFrame(rows)


def credit_sensitivity_table(times: np.ndarray, exposure_by_set: dict[str, np.ndarray], discount: np.ndarray,
                             credits: dict, spreads_bp: tuple[float, ...], recoveries: tuple[float, ...]) -> pd.DataFrame:
    """CVA under alternative credit inputs, holding the simulated exposures fixed.

    Three views, each with everything else at its base value:
      * spread        - assumed spread varies, recovery fixed, hazard re-implied (lambda ~ s / (1 - R))
      * recovery_hazard_fixed - recovery varies, hazard fixed: only LGD changes, so CVA is linear in 1 - R
      * recovery_spread_fixed - recovery varies, spread fixed: hazard re-implied, so LGD x PD barely moves
    """
    def cva(exposure: np.ndarray, hazard: float, lgd: float) -> float:
        return estimate_cva(exposure, discount, HazardCurve.flat(hazard).marginal_pd(times), lgd).value

    rows = []
    for ns, credit in credits.items():
        exposure = exposure_by_set[ns]
        for s_bp in spreads_bp:
            rows.append({"netting_set_id": ns, "view": "spread", "input": s_bp,
                         "cva": cva(exposure, hazard_from_spread(s_bp / 1e4, credit.recovery), credit.lgd)})
        for recovery in recoveries:
            rows.append({"netting_set_id": ns, "view": "recovery_hazard_fixed", "input": recovery,
                         "cva": cva(exposure, credit.hazard, 1.0 - recovery)})
            rows.append({"netting_set_id": ns, "view": "recovery_spread_fixed", "input": recovery,
                         "cva": cva(exposure, hazard_from_spread(credit.spread, recovery), 1.0 - recovery)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase F: convergence and market sensitivities
# ---------------------------------------------------------------------------

def headline_per_path(curve: DiscountCurve, portfolio: list[Swap], credits: dict, params: HullWhiteParams,
                      n_paths: int, seed: int, step: float = 1 / 12) -> dict[str, dict[str, np.ndarray]]:
    """Simulate, revalue and return per-path headline CVA and per-path EPE for each netting set.

    Headline CVA uses pathwise discounting on the full grid, exactly as in run_all.py stage D;
    the per-path EPE is the time-average of each path's netted exposure over the set's life.
    Both are linear in exposure, so their path averages are the estimates and their
    sample standard deviations give the Monte Carlo error.
    """
    grid = build_time_grid(portfolio, step=step)
    paths = simulate_paths(params, grid, n_paths, np.random.default_rng(seed))
    netted = netted_exposure(revalue_portfolio(portfolio, curve, paths), portfolio)
    discount = paths.discount_factors(curve)
    horizons = netting_set_horizons(portfolio)
    out = {}
    for ns, credit in credits.items():
        inside = grid <= horizons[ns] + 1e-9
        out[ns] = {
            "cva": pathwise_cva(netted[ns], discount, credit.hazard_curve().marginal_pd(grid), credit.lgd),
            "epe": np.trapezoid(netted[ns][:, inside], grid[inside], axis=1) / horizons[ns],
        }
    return out


def _mean_se(samples: np.ndarray) -> tuple[float, float]:
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size))


def convergence_table(curve: DiscountCurve, portfolio: list[Swap], credits: dict, params: HullWhiteParams,
                      settings: SimulationSettings, path_counts: tuple[int, ...], seed_stride: int = 1_000) -> pd.DataFrame:
    """CVA and EPE estimates with standard errors and 95% intervals as the number of paths grows.

    Paths come in independent chunks of ``settings.n_paths``; chunk k uses seed
    ``settings.seed + seed_stride * k``, so chunk 0 is the base case. The estimate at N
    paths uses the first N pooled paths (nested samples).
    """
    counts = sorted(n for n in path_counts if n >= 2)
    n_chunks = int(np.ceil(max(counts) / settings.n_paths))
    chunks = [headline_per_path(curve, portfolio, credits, params, settings.n_paths, settings.seed + seed_stride * k,
                                settings.time_step) for k in range(n_chunks)]
    pooled = {ns: {key: np.concatenate([c[ns][key] for c in chunks]) for key in ("cva", "epe")} for ns in credits}
    pooled_total = sum(pooled[ns]["cva"] for ns in credits)
    rows = []
    for n in counts:
        for ns, series in [*((ns, pooled[ns]) for ns in credits), ("TOTAL", {"cva": pooled_total})]:
            cva, cva_se = _mean_se(series["cva"][:n])
            row = {"netting_set_id": ns, "n_paths": n, "cva": cva, "cva_se": cva_se,
                   "ci95_low": cva - 1.96 * cva_se, "ci95_high": cva + 1.96 * cva_se}
            if "epe" in series:
                row["epe"], row["epe_se"] = _mean_se(series["epe"][:n])
            rows.append(row)
    return pd.DataFrame(rows)


def scale_maturities(portfolio: list[Swap], curve: DiscountCurve, factor: float) -> list[Swap]:
    """Stretch every trade's tenor by ``factor`` (rounded to a quarter) and re-strike it.

    The new fixed rate is the new par rate plus the trade's original offset to par, so the
    maturity effect is not mixed up with a change in moneyness.
    """
    if factor == 1.0:
        return list(portfolio)
    scaled = []
    for swap in portfolio:
        offset = swap.fixed_rate - par_rate(swap, curve)
        tenor = max(round((swap.maturity - swap.start) * factor * 4) / 4, 0.25)
        template = replace(swap, maturity=swap.start + tenor)
        scaled.append(replace(template, fixed_rate=par_rate(template, curve) + offset))
    return scaled


def market_sensitivity_table(curve: DiscountCurve, portfolio: list[Swap], credits: dict, params: HullWhiteParams,
                             settings: SimulationSettings, sigmas: tuple[float, ...], mean_reversions: tuple[float, ...],
                             maturity_scales: tuple[float, ...], notional_scales: tuple[float, ...]) -> pd.DataFrame:
    """Headline CVA under alternative market and portfolio inputs, everything else at base.

    Every run re-simulates with the base seed (common random numbers): sigma changes on the
    same grid reuse the same shocks exactly, and the other runs share them as far as the
    grid allows. Notional scaling is exact (values are linear in notional), so it reuses
    the base run.
    """
    def run(pf: list[Swap], hw: HullWhiteParams) -> dict[str, np.ndarray]:
        per_path = headline_per_path(curve, pf, credits, hw, settings.n_paths, settings.seed, settings.time_step)
        out = {ns: per_path[ns]["cva"] for ns in credits}
        out["TOTAL"] = sum(out[ns] for ns in credits)
        return out

    base = run(portfolio, params)
    scenarios: list[tuple[str, float, dict[str, np.ndarray]]] = []
    for sigma in sigmas:
        scenarios.append(("sigma", sigma, base if np.isclose(sigma, params.volatility) else
                          run(portfolio, replace(params, volatility=sigma))))
    for a in mean_reversions:
        scenarios.append(("mean_reversion", a, base if np.isclose(a, params.mean_reversion) else
                          run(portfolio, replace(params, mean_reversion=a))))
    for factor in maturity_scales:
        scenarios.append(("maturity_scale", factor, base if factor == 1.0 else run(scale_maturities(portfolio, curve, factor), params)))
    for factor in notional_scales:
        scenarios.append(("notional_scale", factor, {k: factor * v for k, v in base.items()}))

    rows = []
    for parameter, value, result in scenarios:
        for ns, per_path in result.items():
            cva, se = _mean_se(per_path)
            base_cva = float(base[ns].mean())
            rows.append({"parameter": parameter, "value": value, "netting_set_id": ns, "cva": cva, "cva_se": se,
                         "change_vs_base": cva / base_cva - 1.0 if base_cva else 0.0})
    return pd.DataFrame(rows)


def tornado_table(market: pd.DataFrame, times: np.ndarray, netted: dict[str, np.ndarray], discount: np.ndarray,
                  credits: dict, spread_factors: tuple[float, float] = (0.5, 1.5),
                  recoveries: tuple[float, float] = (0.2, 0.6),
                  market_ranges: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Portfolio CVA at a low and a high value of each input, everything else at base.

    Market rows are read from ``market_sensitivity_table`` output. Credit rows reuse the
    base exposures: the spread is scaled for every counterparty (hazard re-implied as
    s / (1 - R)), and recovery moves with the hazard held fixed, so only LGD changes.
    """
    market_ranges = market_ranges or {"sigma": (0.005, 0.015), "mean_reversion": (0.01, 0.10),
                                      "maturity_scale": (0.75, 1.25), "notional_scale": (0.5, 1.5)}
    total = market[market["netting_set_id"] == "TOTAL"]
    base = float(total.loc[(total["parameter"] == "notional_scale") & np.isclose(total["value"], 1.0), "cva"].iloc[0])

    def total_cva(hazard_of, lgd_of) -> float:
        return sum(estimate_cva(netted[ns], discount, HazardCurve.flat(hazard_of(c)).marginal_pd(times), lgd_of(c)).value
                   for ns, c in credits.items())

    rows = []
    for factor in spread_factors:
        rows.append(("spread_scale", factor, total_cva(lambda c: hazard_from_spread(factor * c.spread, c.recovery), lambda c: c.lgd)))
    for recovery in recoveries:
        rows.append(("recovery_hazard_fixed", recovery, total_cva(lambda c: c.hazard, lambda c: 1.0 - recovery)))
    for parameter, (low, high) in market_ranges.items():
        for value in (low, high):
            match = total[(total["parameter"] == parameter) & np.isclose(total["value"], value)]
            rows.append((parameter, value, float(match["cva"].iloc[0])))
    frame = pd.DataFrame(rows, columns=["parameter", "value", "cva"])
    frame["change_vs_base"] = frame["cva"] / base - 1.0
    frame["base_cva"] = base
    return frame
