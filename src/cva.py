"""Unilateral Credit Valuation Adjustment (CVA).

CVA is the discounted expected loss from the counterparty defaulting while the
netting set has positive value to us. With default independent of rates:

    CVA = LGD * int_0^T E[D(0,t) E(t)] dPD(t)
        ~ LGD * sum_i  dEE_i * dPD_i,      dPD_i = S(t_{i-1}) - S(t_i)

where dEE_i is the discounted expected exposure averaged over the interval's two
endpoints (trapezoid). The sum runs over every interval of the full simulation grid
(monthly plus every cash-flow date); yearly buckets are only an aggregation of those
interval contributions for presentation.

* Headline: discount pathwise with the simulated bank account D(0, t), so
  dEE(t) = E[D(0,t) E(t)]. This is the risk-neutral expectation the formula calls for,
  because exposure and discount factors are correlated under stochastic rates.
* Benchmark: the deterministic approximation of spec section 7, DF(t) = P(0, t) from
  today's curve, so dEE(t) = P(0,t) EE(t). It differs from the headline by the
  covariance between rates and exposure, and equals it exactly when sigma = 0.

Because the estimator is linear in the exposure, CVA is the average over paths of a
pathwise CVA. That average gives the Monte Carlo standard error directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CvaEstimate:
    """A Monte Carlo CVA estimate with its standard error."""

    value: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.value - 1.96 * self.std_error, self.value + 1.96 * self.std_error


def _bucket_average(values: np.ndarray) -> np.ndarray:
    """Average of each interval's two endpoints along the last axis."""
    return 0.5 * (values[..., 1:] + values[..., :-1])


def pathwise_cva(exposure: np.ndarray, discount: np.ndarray, marginal_pd: np.ndarray, lgd: float) -> np.ndarray:
    """CVA contribution of each path: LGD * sum_i avg_i(D * E) * dPD_i.

    ``exposure`` is (n_paths, n_times); ``discount`` is either today's P(0, t) with shape
    (n_times,) or pathwise D(0, t) with shape (n_paths, n_times); ``marginal_pd`` has
    length n_times - 1.
    """
    return lgd * (_bucket_average(exposure * discount) @ marginal_pd)


def estimate_cva(exposure: np.ndarray, discount: np.ndarray, marginal_pd: np.ndarray, lgd: float) -> CvaEstimate:
    """CVA = mean of the pathwise CVAs, with standard error std / sqrt(n)."""
    per_path = pathwise_cva(exposure, discount, marginal_pd, lgd)
    n = per_path.size
    std_error = float(per_path.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return CvaEstimate(float(per_path.mean()), std_error, n)


def discounted_ee(exposure: np.ndarray, discount: np.ndarray) -> np.ndarray:
    """dEE(t) = E[D(0,t) E(t)] (pathwise D, headline) or P(0,t) EE(t) (curve DF, benchmark)."""
    return (exposure * discount).mean(axis=0)


def cva_by_bucket(discounted_expected_exposure: np.ndarray, marginal_pd: np.ndarray, lgd: float) -> np.ndarray:
    """CVA from each grid interval (t_{i-1}, t_i]: LGD * avg_i(dEE) * dPD_i. Sums to the total CVA."""
    return lgd * _bucket_average(discounted_expected_exposure) * marginal_pd


def aggregate_buckets(times: np.ndarray, by_interval: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum interval contributions into coarser buckets (e.g. yearly), keyed by each interval's end date."""
    interval_end = times[1:]
    which = np.clip(np.searchsorted(edges, interval_end - 1e-9, side="left") - 1, 0, edges.size - 2)
    return np.bincount(which, weights=by_interval, minlength=edges.size - 1)


def cva_summary_row(netting_set_id: str, counterparty_id: str, cva_netted: CvaEstimate, cva_gross: CvaEstimate,
                    cva_benchmark: CvaEstimate, spread: float, recovery: float, hazard: float, pd_horizon: float,
                    n_intervals: int) -> dict:
    """One reporting row for a netting set.

    ``cva_netted`` / ``cva_gross`` are the headline (pathwise-discounted) estimates;
    ``cva_benchmark`` is the deterministic P(0,t) x EE approximation for the netted set.
    """
    return {
        "netting_set_id": netting_set_id,
        "counterparty_id": counterparty_id,
        "spread_bp": spread * 1e4,
        "recovery": recovery,
        "hazard": hazard,
        "pd_to_horizon": pd_horizon,
        "cva_netted": cva_netted.value,
        "cva_netted_se": cva_netted.std_error,
        "cva_gross": cva_gross.value,
        "netting_reduction": 1.0 - cva_netted.value / cva_gross.value if cva_gross.value > 0 else 0.0,
        "cva_benchmark_curve_df": cva_benchmark.value,
        "benchmark_vs_headline": cva_benchmark.value / cva_netted.value - 1.0 if cva_netted.value > 0 else 0.0,
        "integration_intervals": n_intervals,
    }


def summary_frame(rows: list[dict], total_netted: CvaEstimate) -> pd.DataFrame:
    """Summary table with a portfolio total row (CVA adds across netting sets).

    ``total_netted`` is estimated from the per-path sum over netting sets: the sets
    share the same simulated paths, so their errors are correlated and cannot be
    added in quadrature.
    """
    frame = pd.DataFrame(rows)
    total = {"netting_set_id": "TOTAL", "counterparty_id": "",
             "cva_netted": total_netted.value,
             "cva_netted_se": total_netted.std_error,
             "cva_gross": frame["cva_gross"].sum(),
             "cva_benchmark_curve_df": frame["cva_benchmark_curve_df"].sum(),
             "integration_intervals": frame["integration_intervals"].max()}
    total["netting_reduction"] = 1.0 - total["cva_netted"] / total["cva_gross"]
    total["benchmark_vs_headline"] = total["cva_benchmark_curve_df"] / total["cva_netted"] - 1.0
    return pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
