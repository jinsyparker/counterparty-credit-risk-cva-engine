"""Simplified one-way CSA collateral and collateralized exposure.

The counterparty posts collateral to us when the netting set is worth more than the
threshold to us; we never post (consistent with unilateral CVA). If the counterparty
defaults at t, the last collateral call that was met is assumed to be at t - MPoR (the
margin period of risk), so:

    collateral held       C(t)   = max(V(t - MPoR) - H - MTA, 0)
    collateralized exp.   E_c(t) = max(V(t) - C(t), 0)

with V the netted value to us, H the threshold and MTA the minimum transfer amount.
For t < MPoR the look-back uses V(0): the CSA is in place, and collateralized, from
inception.

Simplifications (stated, not hidden):
* MTA is added to the threshold, a common static stand-in for the path-dependent rule
  "only transfer when the call exceeds the MTA".
* Both parties keep paying contractual cash flows during the MPoR. A coupon we pay
  inside the window therefore raises exposure: the counterparty holds collateral sized
  before the payment. This shows up as spikes on payment dates.
* No independent amount, initial margin, haircuts, rehypothecation or disputes.

Properties used in the tests: C >= 0, so E_c <= max(V, 0); E_c rises with H and MTA
on every path; H = infinity gives the uncollateralized exposure exactly; H = 0 with
MPoR = 0 gives zero exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cva import estimate_cva
from src.exposure import (
    expected_exposure,
    expected_positive_exposure,
    gross_exposure,
    netting_set_horizons,
    potential_future_exposure,
)
from src.simulation import netting_set_mtm, refine_paths, revalue_portfolio, time_index

TWO_WEEKS = 14 / 365


@dataclass(frozen=True)
class CsaTerms:
    """One collateral scenario. ``threshold = inf`` means uncollateralized."""

    name: str
    threshold: float = np.inf
    minimum_transfer: float = 0.0
    mpor: float = TWO_WEEKS

    def __post_init__(self) -> None:
        if self.threshold < 0 or self.minimum_transfer < 0 or self.mpor < 0:
            raise ValueError("threshold, minimum_transfer and mpor must be non-negative")

    @property
    def collateralized(self) -> bool:
        return np.isfinite(self.threshold)

    @property
    def effective_threshold(self) -> float:
        """H + MTA: the netted value above which collateral is held."""
        return self.threshold + self.minimum_transfer


def lookback_dates(times: np.ndarray, mpor: float) -> np.ndarray:
    """max(t - MPoR, 0) for each exposure date: when the last met collateral call happened."""
    return np.round(np.maximum(np.asarray(times, dtype=float) - mpor, 0.0), 10)


def collateral_balance(lagged_values: np.ndarray, terms: CsaTerms) -> np.ndarray:
    """C(t) = max(V(t - MPoR) - H - MTA, 0): one-way, posted by the counterparty only."""
    if not terms.collateralized:
        return np.zeros_like(lagged_values)
    return np.maximum(lagged_values - terms.effective_threshold, 0.0)


def collateralized_exposure(values: np.ndarray, lagged_values: np.ndarray, terms: CsaTerms) -> np.ndarray:
    """E_c(t) = max(V(t) - C(t), 0) on every path and exposure date."""
    return np.maximum(values - collateral_balance(lagged_values, terms), 0.0)


# ---------------------------------------------------------------------------
# Analysis shared by run_all.py and the notebooks
# ---------------------------------------------------------------------------

def collateral_analysis(curve, portfolio, paths, credits: dict, scenarios: tuple[CsaTerms, ...],
                        mpor_days: tuple[int, ...], bridge_rng: np.random.Generator, pfe_quantile: float = 0.95) -> dict:
    """Exposure and CVA for every netting set under each CSA scenario, on the given base paths.

    The look-back dates t - MPoR are added to ``paths`` with an exact bridge, so the base
    dates (and therefore the uncollateralized numbers) are exactly those of ``paths``.
    CVA uses the headline pathwise discounting and the full base grid, as in Phase D.

    Returns DataFrames ``summary`` (netting set x scenario), ``mitigation`` (gross ->
    netted -> netted + CSA), ``mpor`` (zero-threshold MPoR sensitivity) and ``profiles``
    (EE / PFE through time), plus ``base_dates_unchanged`` (a consistency flag).
    """
    grid = paths.times
    mpors = sorted({s.mpor for s in scenarios if s.collateralized} | {d / 365 for d in mpor_days})
    lookbacks = {m: lookback_dates(grid, m) for m in mpors}
    fine_grid = np.unique(np.concatenate([grid, *lookbacks.values()]))
    fine = refine_paths(paths, fine_grid, bridge_rng)
    fine_cube = revalue_portfolio(portfolio, curve, fine)
    primary = time_index(fine_grid, grid)
    base_x_unchanged = bool(np.array_equal(fine.x[:, primary], paths.x))
    values = netting_set_mtm(fine_cube, portfolio)
    now = {ns: v[:, primary] for ns, v in values.items()}
    lagged = {(ns, m): values[ns][:, time_index(fine_grid, lookbacks[m])] for ns in values for m in mpors}
    gross = gross_exposure(fine_cube[:, :, primary], portfolio)
    discount = paths.discount_factors(curve)
    horizons = netting_set_horizons(portfolio)

    summary, mitigation, mpor_rows, profiles = [], [], [], []
    for ns, credit in credits.items():
        mpd = credit.hazard_curve().marginal_pd(grid)

        def cva(exposure):
            return estimate_cva(exposure, discount, mpd, credit.lgd)

        gross_cva = cva(gross[ns])
        mitigation.append({"netting_set_id": ns, "stage": "Gross (no netting, no collateral)",
                           "cva": gross_cva.value, "cva_se": gross_cva.std_error})
        base_cva = None
        for terms in scenarios:
            lag = lagged[(ns, terms.mpor)] if terms.collateralized else now[ns]
            exposure = collateralized_exposure(now[ns], lag, terms)
            ee, pfe = expected_exposure(exposure), potential_future_exposure(exposure, pfe_quantile)
            estimate = cva(exposure)
            base_cva = estimate.value if base_cva is None else base_cva
            summary.append({
                "netting_set_id": ns, "scenario": terms.name,
                "threshold": terms.threshold, "minimum_transfer": terms.minimum_transfer,
                "mpor_days": round(terms.mpor * 365) if terms.collateralized else np.nan,
                "epe_time_avg_ee": expected_positive_exposure(grid, ee, horizons[ns]),
                "peak_ee": ee.max(), "peak_pfe": pfe.max(),
                "cva": estimate.value, "cva_se": estimate.std_error,
                "cva_reduction_vs_uncollateralized": 1.0 - estimate.value / base_cva if base_cva else 0.0,
            })
            stage = "Netted, uncollateralized" if not terms.collateralized else f"Netted + CSA: {terms.name}"
            mitigation.append({"netting_set_id": ns, "stage": stage, "cva": estimate.value, "cva_se": estimate.std_error})
            profiles.append(pd.DataFrame({"netting_set_id": ns, "scenario": terms.name, "time": grid, "ee": ee, "pfe": pfe}))
        for days in mpor_days:
            m = days / 365
            exposure = collateralized_exposure(now[ns], lagged[(ns, m)], CsaTerms("zero", 0.0, 0.0, m))
            estimate = cva(exposure)
            mpor_rows.append({"netting_set_id": ns, "mpor_days": days,
                              "epe_time_avg_ee": expected_positive_exposure(grid, expected_exposure(exposure), horizons[ns]),
                              "cva": estimate.value, "cva_se": estimate.std_error,
                              "cva_share_of_uncollateralized": estimate.value / base_cva if base_cva else 0.0})

    mitigation = pd.DataFrame(mitigation)
    gross_by_set = mitigation[mitigation["stage"].str.startswith("Gross")].set_index("netting_set_id")["cva"]
    mitigation["reduction_vs_gross"] = 1.0 - mitigation["cva"] / mitigation["netting_set_id"].map(gross_by_set)
    return {
        "summary": pd.DataFrame(summary), "mitigation": mitigation, "mpor": pd.DataFrame(mpor_rows),
        "profiles": pd.concat(profiles, ignore_index=True), "fine_grid_size": fine_grid.size,
        "base_dates_unchanged": base_x_unchanged,
    }
