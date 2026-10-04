"""Vanilla interest-rate swap pricer (single-curve).

The core leg formulas take arrays of discount factors rather than a curve, with
cash flows on the last axis. The same functions therefore price a swap today
(discount factors from the initial curve, shape ``(n_cf,)``) and on simulated
future states (discount factors P(t, T_i) per path, shape ``(n_paths, n_cf)``).

Formulas (notional N, accruals tau_i, discount factors P_i):

    annuity        A          = sum_i tau_i P(T_i)
    fixed leg      PV_fixed   = N K A
    forward rate   F_j        = (P(S_j) / P(T_j) - 1) / tau_j
    floating leg   PV_float   = N sum_j tau_j F_j P(T_j)   (= N (P(T_0) - P(T_n)) single-curve)
    par rate       K_par      = PV_float / (N A)
    swap value     V          = sign * (PV_float - PV_fixed),  sign = +1 payer, -1 receiver
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.curves import DiscountCurve
from src.instruments import Swap


# ---------------------------------------------------------------------------
# Core formulas on discount-factor arrays (cash flows on the last axis)
# ---------------------------------------------------------------------------

def annuity_from_dfs(accruals: np.ndarray, pay_dfs: np.ndarray) -> np.ndarray:
    """A = sum_i tau_i P(T_i)."""
    return np.sum(accruals * pay_dfs, axis=-1)


def fixed_leg_from_dfs(notional: float, fixed_rate: float, accruals: np.ndarray, pay_dfs: np.ndarray) -> np.ndarray:
    """PV_fixed = N K sum_i tau_i P(T_i)."""
    return notional * fixed_rate * annuity_from_dfs(accruals, pay_dfs)


def forward_rates_from_dfs(accruals: np.ndarray, start_dfs: np.ndarray, pay_dfs: np.ndarray) -> np.ndarray:
    """Simple forward rate for each period: F_j = (P(S_j) / P(T_j) - 1) / tau_j."""
    return (start_dfs / pay_dfs - 1.0) / accruals


def float_leg_from_dfs(notional: float, accruals: np.ndarray, forwards: np.ndarray, pay_dfs: np.ndarray) -> np.ndarray:
    """PV_float = N sum_j tau_j F_j P(T_j).

    ``forwards`` is passed in so a period whose rate has already been fixed can use
    its fixing instead of the forward rate (needed once trades are revalued in the
    future).
    """
    return notional * np.sum(accruals * forwards * pay_dfs, axis=-1)


def swap_value_from_legs(sign: float, pv_float: np.ndarray, pv_fixed: np.ndarray) -> np.ndarray:
    """V = sign * (PV_float - PV_fixed): received leg minus paid leg."""
    return sign * (pv_float - pv_fixed)


# ---------------------------------------------------------------------------
# Time-0 pricing against the initial curve
# ---------------------------------------------------------------------------

def annuity(swap: Swap, curve: DiscountCurve) -> float:
    """Fixed-leg annuity per unit notional, sum_i tau_i P(0, T_i)."""
    sched = swap.fixed_schedule
    return float(annuity_from_dfs(sched.accruals, curve.df(sched.pay_times)))


def pv_fixed_leg(swap: Swap, curve: DiscountCurve) -> float:
    """Present value of the fixed leg (positive number)."""
    sched = swap.fixed_schedule
    return float(fixed_leg_from_dfs(swap.notional, swap.fixed_rate, sched.accruals, curve.df(sched.pay_times)))


def pv_float_leg(swap: Swap, curve: DiscountCurve) -> float:
    """Present value of the floating leg (positive number)."""
    sched = swap.float_schedule
    pay_dfs = curve.df(sched.pay_times)
    forwards = forward_rates_from_dfs(sched.accruals, curve.df(sched.accrual_start), pay_dfs)
    return float(float_leg_from_dfs(swap.notional, sched.accruals, forwards, pay_dfs))


def par_rate(swap: Swap, curve: DiscountCurve) -> float:
    """Fixed rate that makes the swap worth zero: PV_float / (N A)."""
    return pv_float_leg(swap, curve) / (swap.notional * annuity(swap, curve))


def swap_mtm(swap: Swap, curve: DiscountCurve) -> float:
    """Mark-to-market from our perspective."""
    return float(swap_value_from_legs(swap.sign, pv_float_leg(swap, curve), pv_fixed_leg(swap, curve)))


def cashflow_table(swap: Swap, curve: DiscountCurve) -> pd.DataFrame:
    """Period-by-period cash flows of both legs, signed from our perspective.

    Floating cash flows are projected at the forward rate. The ``pv`` column sums to
    the swap MtM.
    """
    rows = []
    fixed, flt = swap.fixed_schedule, swap.float_schedule
    float_forwards = curve.simple_forward(flt.accrual_start, flt.pay_times)
    legs = [
        ("fixed", fixed, np.full(fixed.pay_times.size, swap.fixed_rate), -swap.sign),
        ("float", flt, float_forwards, swap.sign),
    ]
    for leg, sched, rates, leg_sign in legs:
        dfs = curve.df(sched.pay_times)
        cashflows = leg_sign * swap.notional * rates * sched.accruals
        for i in range(sched.pay_times.size):
            rows.append(
                {
                    "leg": leg,
                    "accrual_start": sched.accrual_start[i],
                    "pay_time": sched.pay_times[i],
                    "accrual": sched.accruals[i],
                    "rate": rates[i],
                    "cashflow": cashflows[i],
                    "discount_factor": dfs[i],
                    "pv": cashflows[i] * dfs[i],
                }
            )
    return pd.DataFrame(rows).sort_values(["pay_time", "leg"]).reset_index(drop=True)


def price_portfolio(portfolio: list[Swap], curve: DiscountCurve) -> pd.DataFrame:
    """Time-0 valuation of every trade: leg PVs, par rate and MtM."""
    rows = []
    for swap in portfolio:
        rows.append(
            {
                "trade_id": swap.trade_id,
                "counterparty_id": swap.counterparty_id,
                "netting_set_id": swap.netting_set_id,
                "direction": swap.direction,
                "notional": swap.notional,
                "start": swap.start,
                "maturity": swap.maturity,
                "fixed_rate": swap.fixed_rate,
                "par_rate": par_rate(swap, curve),
                "pv_fixed": pv_fixed_leg(swap, curve),
                "pv_float": pv_float_leg(swap, curve),
                "mtm": swap_mtm(swap, curve),
            }
        )
    return pd.DataFrame(rows)


def netting_set_summary(valuation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a ``price_portfolio`` table by netting set.

    ``net_mtm`` is the legally netted value; ``gross_positive_mtm`` is what would be at
    risk without netting (sum of trade-level positive values).
    """
    grouped = valuation.groupby(["netting_set_id", "counterparty_id"], sort=True)
    return grouped.agg(
        n_trades=("trade_id", "count"),
        net_mtm=("mtm", "sum"),
        gross_positive_mtm=("mtm", lambda v: v.clip(lower=0).sum()),
        gross_notional=("notional", "sum"),
    ).reset_index()
