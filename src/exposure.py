"""Counterparty exposure from simulated portfolio values.

Definitions (V = value to us on one path and date; E = exposure):

    positive exposure     E(t)      = max(V(t), 0)
    netted exposure       E_net(t)  = max(sum_{i in netting set} V_i(t), 0)
    gross exposure        E_gross(t)= sum_{i in netting set} max(V_i(t), 0)
    expected exposure     EE(t)     = mean over paths of E(t)
    potential future exp. PFE_q(t)  = q-quantile over paths of E(t)
    EPE (time-average EE)  EPE     = (1/T) int_0^T EE(t) dt   (trapezoid on the grid, T = netting set's life)

Netting is applied to trade values on each path and date *before* the positive
part, so E_net <= E_gross pathwise (the positive part is subadditive).

"EPE" here always means the time-average of EE over the netting set's whole life.
It is not the regulatory Effective EPE (EEPE), which averages the running maximum of
EE over the first year only.

PFE uses the empirical inverse CDF: the smallest simulated exposure e with
P_n(E <= e) >= q. That makes it an actual simulated value that reconciles exactly
with the simulated distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.instruments import Swap
from src.simulation import netting_set_mtm


def positive_exposure(values: np.ndarray) -> np.ndarray:
    """E = max(V, 0), elementwise."""
    return np.maximum(values, 0.0)


def netted_exposure(cube: np.ndarray, portfolio: list[Swap]) -> dict[str, np.ndarray]:
    """{netting_set_id: max(sum of trade values, 0)}, each (n_paths, n_times)."""
    return {ns: positive_exposure(mtm) for ns, mtm in netting_set_mtm(cube, portfolio).items()}


def gross_exposure(cube: np.ndarray, portfolio: list[Swap]) -> dict[str, np.ndarray]:
    """{netting_set_id: sum of trade-level positive exposures} (no netting benefit)."""
    return netting_set_mtm(positive_exposure(cube), portfolio)


def expected_exposure(exposure: np.ndarray) -> np.ndarray:
    """EE(t): mean over paths (axis 0) of an (n_paths, n_times) exposure array."""
    return exposure.mean(axis=0)


def potential_future_exposure(exposure: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """PFE_q(t): empirical q-quantile over paths (inverse-CDF definition)."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    return np.quantile(exposure, quantile, axis=0, method="inverted_cdf")


def expected_positive_exposure(times: np.ndarray, ee: np.ndarray, horizon: float | None = None) -> float:
    """EPE as the time-average EE, (1/T) int_0^T EE(t) dt (trapezoid); T defaults to the last grid date. Not Effective EPE."""
    horizon = times[-1] if horizon is None else horizon
    inside = times <= horizon + 1e-9
    return float(np.trapezoid(ee[inside], times[inside]) / horizon)


def netting_set_horizons(portfolio: list[Swap]) -> dict[str, float]:
    """Last maturity of each netting set (the life over which its EPE is averaged)."""
    horizons: dict[str, float] = {}
    for swap in portfolio:
        horizons[swap.netting_set_id] = max(horizons.get(swap.netting_set_id, 0.0), swap.maturity)
    return horizons


def profile_table(times: np.ndarray, netted: dict[str, np.ndarray], gross: dict[str, np.ndarray],
                  quantile: float = 0.95) -> pd.DataFrame:
    """Long table of EE and PFE through time, netted and gross, for every netting set."""
    frames = []
    for ns in netted:
        frames.append(pd.DataFrame({
            "netting_set_id": ns,
            "time": times,
            "ee_netted": expected_exposure(netted[ns]),
            "pfe_netted": potential_future_exposure(netted[ns], quantile),
            "ee_gross": expected_exposure(gross[ns]),
            "pfe_gross": potential_future_exposure(gross[ns], quantile),
        }))
    return pd.concat(frames, ignore_index=True)


def exposure_summary(times: np.ndarray, netted: dict[str, np.ndarray], gross: dict[str, np.ndarray],
                     horizons: dict[str, float], quantile: float = 0.95) -> pd.DataFrame:
    """One row per netting set: EPE (netted vs gross), netting reduction, peak EE and peak PFE."""
    rows = []
    for ns in netted:
        ee_net, ee_gross = expected_exposure(netted[ns]), expected_exposure(gross[ns])
        pfe_net = potential_future_exposure(netted[ns], quantile)
        epe_net = expected_positive_exposure(times, ee_net, horizons[ns])
        epe_gross = expected_positive_exposure(times, ee_gross, horizons[ns])
        rows.append({
            "netting_set_id": ns,
            "horizon": horizons[ns],
            "epe_netted": epe_net,
            "epe_gross": epe_gross,
            "netting_reduction": 1.0 - epe_net / epe_gross if epe_gross > 0 else 0.0,
            "peak_ee_netted": ee_net.max(),
            "peak_ee_time": times[ee_net.argmax()],
            f"peak_pfe{quantile * 100:g}_netted": pfe_net.max(),
            "peak_pfe_time": times[pfe_net.argmax()],
        })
    return pd.DataFrame(rows)
