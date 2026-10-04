"""Vanilla fixed-for-floating interest-rate swap records and portfolio helpers.

Dates are year fractions from the valuation date: ``start`` is the effective date and
``maturity`` the final payment date (both absolute times, not tenors). Schedules are
regular periods of length 1/freq generated backward from maturity, with a short first
period if needed; accrual fractions are plain time differences (no day-count or
business-day adjustments).

Direction refers to the fixed leg, from our (the bank's) perspective:
``payer`` = we pay fixed and receive floating, ``receiver`` = the opposite.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTIONS = {"payer": 1.0, "receiver": -1.0}


@dataclass(frozen=True)
class Schedule:
    """Accrual periods of one swap leg."""

    accrual_start: np.ndarray
    pay_times: np.ndarray

    @property
    def accruals(self) -> np.ndarray:
        return self.pay_times - self.accrual_start


def make_schedule(start: float, maturity: float, freq: int) -> Schedule:
    """Regular schedule from ``start`` to ``maturity`` with ``freq`` payments a year."""
    if not 0 <= start < maturity:
        raise ValueError("need 0 <= start < maturity")
    if freq <= 0:
        raise ValueError("freq must be positive")
    n_periods = int(np.ceil((maturity - start) * freq - 1e-9))
    pay_times = np.round(maturity - np.arange(n_periods)[::-1] / freq, 12)
    accrual_start = np.concatenate([[start], pay_times[:-1]])
    return Schedule(accrual_start=accrual_start, pay_times=pay_times)


@dataclass(frozen=True)
class Swap:
    """A vanilla fixed-for-floating interest-rate swap."""

    trade_id: str
    counterparty_id: str
    netting_set_id: str
    notional: float
    fixed_rate: float
    direction: str
    start: float
    maturity: float
    fixed_freq: int = 2
    float_freq: int = 4

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {self.direction!r}")
        if self.notional <= 0:
            raise ValueError("notional must be positive; use direction for the sign")
        make_schedule(self.start, self.maturity, self.fixed_freq)  # validates dates

    @property
    def sign(self) -> float:
        """+1 for a payer swap (value = float - fixed), -1 for a receiver swap."""
        return DIRECTIONS[self.direction]

    @property
    def fixed_schedule(self) -> Schedule:
        return make_schedule(self.start, self.maturity, self.fixed_freq)

    @property
    def float_schedule(self) -> Schedule:
        return make_schedule(self.start, self.maturity, self.float_freq)


PORTFOLIO_COLUMNS = [
    "trade_id",
    "counterparty_id",
    "netting_set_id",
    "notional",
    "fixed_rate",
    "direction",
    "start",
    "maturity",
    "fixed_freq",
    "float_freq",
]


def load_portfolio(path: str | Path) -> list[Swap]:
    """Read swaps from a CSV with one row per trade (see ``PORTFOLIO_COLUMNS``)."""
    table = pd.read_csv(path)
    missing = set(PORTFOLIO_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"portfolio file is missing columns: {sorted(missing)}")
    return [
        Swap(
            trade_id=str(row.trade_id),
            counterparty_id=str(row.counterparty_id),
            netting_set_id=str(row.netting_set_id),
            notional=float(row.notional),
            fixed_rate=float(row.fixed_rate),
            direction=str(row.direction).strip().lower(),
            start=float(row.start),
            maturity=float(row.maturity),
            fixed_freq=int(row.fixed_freq),
            float_freq=int(row.float_freq),
        )
        for row in table.itertuples(index=False)
    ]


def portfolio_to_frame(portfolio: list[Swap]) -> pd.DataFrame:
    """Trade terms as a DataFrame, one row per swap."""
    return pd.DataFrame([{col: getattr(swap, col) for col in PORTFOLIO_COLUMNS} for swap in portfolio])


def group_by_netting_set(portfolio: list[Swap]) -> dict[str, list[Swap]]:
    """Map each netting-set ID to its trades, in portfolio order."""
    groups: dict[str, list[Swap]] = defaultdict(list)
    for swap in portfolio:
        groups[swap.netting_set_id].append(swap)
    return dict(groups)
