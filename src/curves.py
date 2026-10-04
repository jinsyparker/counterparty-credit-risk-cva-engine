"""Discount curve and Treasury par-yield bootstrap.

Conventions
-----------
* Time is a year fraction from the valuation date (t = 0).
* Zero rates are continuously compounded: P(0, t) = exp(-z(t) * t).
* Interpolation is linear in log discount factor, which is the same as holding the
  instantaneous forward rate constant between nodes. It reproduces the nodes
  exactly and keeps P(0, t) positive (and decreasing whenever forwards are positive).
* Before the first node the first zero rate is held flat; after the last node the
  last segment's forward rate is held flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ArrayLike = float | np.ndarray

COUPON_FREQ = 2  # Treasury par yields are semiannual bond-equivalent


def _as_output(values: np.ndarray, like) -> ArrayLike:
    """Return a Python float for scalar input, otherwise the array."""
    return float(values) if np.ndim(like) == 0 else values


@dataclass(frozen=True)
class DiscountCurve:
    """Zero curve P(0, t) = exp(-z(t) t) with log-linear discount-factor interpolation."""

    times: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        zero_rates = np.asarray(self.zero_rates, dtype=float)
        if times.ndim != 1 or times.shape != zero_rates.shape or times.size == 0:
            raise ValueError("times and zero_rates must be non-empty 1-D arrays of equal length")
        if times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "zero_rates", zero_rates)

    @property
    def _log_df_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Knots (0, 0), (t_i, -z_i t_i) of the piecewise-linear log P(0, t)."""
        return np.concatenate([[0.0], self.times]), np.concatenate([[0.0], -self.zero_rates * self.times])

    def log_df(self, t: ArrayLike) -> ArrayLike:
        """log P(0, t): linear between nodes, flat forward beyond the last node."""
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0):
            raise ValueError("t must be non-negative")
        knot_t, knot_y = self._log_df_nodes
        values = np.interp(t_arr, knot_t, knot_y)
        last_forward = (knot_y[-2] - knot_y[-1]) / (knot_t[-1] - knot_t[-2])
        beyond = t_arr > knot_t[-1]
        values = np.where(beyond, knot_y[-1] - last_forward * (t_arr - knot_t[-1]), values)
        return _as_output(values, t)

    def df(self, t: ArrayLike) -> ArrayLike:
        """Discount factor P(0, t)."""
        return _as_output(np.exp(self.log_df(t)), t)

    def zero_rate(self, t: ArrayLike) -> ArrayLike:
        """Continuously compounded zero rate z(t) = -log P(0, t) / t (z(0) = first node rate)."""
        t_arr = np.asarray(t, dtype=float)
        safe_t = np.where(t_arr > 0, t_arr, 1.0)
        values = np.where(t_arr > 0, -np.asarray(self.log_df(safe_t)) / safe_t, self.zero_rates[0])
        return _as_output(values, t)

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> ArrayLike:
        """Continuously compounded forward rate between t1 and t2: log(P(t1)/P(t2)) / (t2 - t1)."""
        t1_arr, t2_arr = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        values = (np.asarray(self.log_df(t1_arr)) - np.asarray(self.log_df(t2_arr))) / (t2_arr - t1_arr)
        return _as_output(values, np.broadcast_arrays(t1, t2)[0])

    def simple_forward(self, t1: ArrayLike, t2: ArrayLike) -> ArrayLike:
        """Simply compounded forward rate F(t1, t2) = (P(t1)/P(t2) - 1) / (t2 - t1)."""
        t1_arr, t2_arr = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        values = (np.exp(np.asarray(self.log_df(t1_arr)) - np.asarray(self.log_df(t2_arr))) - 1.0) / (t2_arr - t1_arr)
        return _as_output(values, np.broadcast_arrays(t1, t2)[0])

    def inst_forward(self, t: ArrayLike) -> ArrayLike:
        """Instantaneous forward f(0, t): piecewise constant, right-continuous at the nodes."""
        t_arr = np.asarray(t, dtype=float)
        knot_t, knot_y = self._log_df_nodes
        segment_forwards = -np.diff(knot_y) / np.diff(knot_t)
        segment = np.clip(np.searchsorted(knot_t, t_arr, side="right") - 1, 0, segment_forwards.size - 1)
        return _as_output(segment_forwards[segment], t)


def par_bond_schedule(maturity: float, freq: int = COUPON_FREQ) -> tuple[np.ndarray, np.ndarray]:
    """Coupon times and accruals of a par bond, generated backward from maturity.

    A tenor that is not a whole number of coupon periods gets a short first period,
    so a 3-month bill is one payment of (1 + y * 0.25) at 0.25.
    """
    n_periods = int(np.ceil(maturity * freq - 1e-9))
    pay_times = np.round(maturity - np.arange(n_periods)[::-1] / freq, 12)
    accruals = np.diff(np.concatenate([[0.0], pay_times]))
    return pay_times, accruals


def bootstrap_par_yields(tenors: np.ndarray, par_yields: np.ndarray, freq: int = COUPON_FREQ) -> DiscountCurve:
    """Bootstrap a zero curve from par yields (decimals) on a semiannual bond-equivalent basis.

    Every quoted yield is treated as the coupon of a bond priced at par:

        1 = y * sum_k tau_k P(0, T_k) + P(0, T_n)

    * Tenors up to one coupon period (bills) have a single payment, so
      P(0, T) = 1 / (1 + y T).
    * Beyond that, par yields are linearly interpolated onto every coupon date and
      the discount factors are solved one coupon date at a time:
      P(0, T_n) = (1 - y_n * sum_{k<n} tau_k P(0, T_k)) / (1 + y_n tau_n).

    The returned curve has a node at every short tenor and every coupon date, so it
    reprices each input bond to par exactly.
    """
    tenors = np.asarray(tenors, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    period = 1.0 / freq

    short = tenors <= period + 1e-12
    node_times = list(tenors[short])
    node_dfs = list(1.0 / (1.0 + par_yields[short] * tenors[short]))
    if not np.isclose(node_times[-1], period):
        raise ValueError("the bootstrap needs a quote at exactly one coupon period (6M for semiannual)")

    df_by_time = dict(zip(np.round(node_times, 12), node_dfs))
    coupon_dates = np.round(np.arange(2, int(round(tenors[-1] * freq)) + 1) * period, 12)
    for maturity in coupon_dates:
        y = np.interp(maturity, tenors, par_yields)
        earlier = np.round(np.arange(1, int(round(maturity * freq))) * period, 12)
        annuity_known = period * sum(df_by_time[t] for t in earlier)
        df_by_time[maturity] = (1.0 - y * annuity_known) / (1.0 + y * period)
        node_times.append(maturity)
        node_dfs.append(df_by_time[maturity])

    times = np.asarray(node_times)
    zero_rates = -np.log(np.asarray(node_dfs)) / times
    return DiscountCurve(times, zero_rates)


def load_par_yields(path: str | Path) -> pd.DataFrame:
    """Read a Treasury par-yield snapshot saved by ``scripts/fetch_treasury_curve.py``."""
    table = pd.read_csv(path).sort_values("tenor_years").reset_index(drop=True)
    table["par_yield"] = table["par_yield_pct"] / 100.0
    return table


def load_treasury_curve(path: str | Path) -> DiscountCurve:
    """Load a par-yield snapshot and bootstrap it into a DiscountCurve."""
    table = load_par_yields(path)
    return bootstrap_par_yields(table["tenor_years"].to_numpy(), table["par_yield"].to_numpy())


def curve_table(curve: DiscountCurve) -> pd.DataFrame:
    """Node-by-node view of the curve: zero rate, discount factor and forward to the next node."""
    times = curve.times
    forwards = curve.inst_forward(times)
    return pd.DataFrame(
        {
            "time": times,
            "zero_rate": curve.zero_rates,
            "discount_factor": curve.df(times),
            "forward_rate_to_next_node": forwards,
        }
    )
