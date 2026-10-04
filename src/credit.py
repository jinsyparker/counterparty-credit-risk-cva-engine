"""Counterparty default probabilities from a hazard-rate (reduced-form) model.

Default arrives at the first jump of a Poisson process with intensity lambda(t):

    survival            S(t)   = exp(-int_0^t lambda(u) du)
    cumulative PD       PD(t)  = 1 - S(t)
    marginal PD         dPD_i  = S(t_{i-1}) - S(t_i)     (default inside (t_{i-1}, t_i])
    credit triangle     lambda ~ s / (1 - R)             (flat spread s, recovery R)
    loss given default  LGD    = 1 - R

The spreads are assumed counterparty spread scenarios (CDS-equivalent spread
assumptions), not market quotes, and lambda ~ s / (1 - R) is an approximation: it
ignores premium accrual, discounting inside the CDS legs and the term structure of
spreads. It is not a calibration to a CDS curve.

The hazard is piecewise constant: lambda_k applies on (t_{k-1}, t_k] and the last
rate is extended flat. The marginal PDs over a grid telescope, so they sum exactly
to PD at the last grid date. Hazard rates are deterministic and independent of
interest rates (no wrong-way risk).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant hazard rate: ``hazards[k]`` applies up to ``times[k]``; the last one continues flat."""

    times: np.ndarray
    hazards: np.ndarray

    def __post_init__(self) -> None:
        times = np.atleast_1d(np.asarray(self.times, dtype=float))
        hazards = np.atleast_1d(np.asarray(self.hazards, dtype=float))
        if times.shape != hazards.shape or times.ndim != 1:
            raise ValueError("times and hazards must be 1-D arrays of equal length")
        if times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        if np.any(hazards < 0):
            raise ValueError("hazard rates must be non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "hazards", hazards)

    @classmethod
    def flat(cls, hazard: float) -> HazardCurve:
        """Constant hazard rate for all horizons."""
        return cls(np.array([1.0]), np.array([hazard]))

    def integrated_hazard(self, t: float | np.ndarray) -> np.ndarray:
        """Lambda(t) = int_0^t lambda(u) du, piecewise linear in t."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("t must be non-negative")
        starts = np.concatenate([[0.0], self.times[:-1]])
        # time spent in each segment; the last segment is open-ended
        ends = np.concatenate([self.times[:-1], [np.inf]])
        spent = np.clip(t[..., None] - starts, 0.0, ends - starts)
        return spent @ self.hazards

    def survival(self, t: float | np.ndarray) -> np.ndarray:
        """S(t) = exp(-Lambda(t))."""
        return np.exp(-self.integrated_hazard(t))

    def cumulative_pd(self, t: float | np.ndarray) -> np.ndarray:
        """PD(t) = 1 - S(t)."""
        return -np.expm1(-self.integrated_hazard(t))

    def marginal_pd(self, grid: np.ndarray) -> np.ndarray:
        """Default probability inside each grid interval: S(t_{i-1}) - S(t_i), length len(grid) - 1."""
        return -np.diff(self.survival(np.asarray(grid, dtype=float)))


def hazard_from_spread(spread: float, recovery: float) -> float:
    """Credit-triangle approximation lambda ~ s / (1 - R); s is an assumed flat CDS-equivalent spread in decimals."""
    if not 0 <= recovery < 1:
        raise ValueError("recovery must be in [0, 1) to imply a hazard rate from a spread")
    return spread / (1.0 - recovery)


@dataclass(frozen=True)
class CounterpartyCredit:
    """Credit inputs for one counterparty / netting set."""

    counterparty_id: str
    netting_set_id: str
    spread: float
    recovery: float

    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery

    @property
    def hazard(self) -> float:
        return hazard_from_spread(self.spread, self.recovery)

    def hazard_curve(self) -> HazardCurve:
        return HazardCurve.flat(self.hazard)


def load_counterparties(path: str | Path) -> dict[str, CounterpartyCredit]:
    """Read ``counterparties.csv`` into {netting_set_id: CounterpartyCredit}; assumed spreads are given in bp."""
    table = pd.read_csv(path)
    return {
        str(row.netting_set_id): CounterpartyCredit(
            counterparty_id=str(row.counterparty_id),
            netting_set_id=str(row.netting_set_id),
            spread=float(row.cds_spread_bp) / 10_000,
            recovery=float(row.recovery_rate),
        )
        for row in table.itertuples(index=False)
    }


def credit_curve_table(credits: dict[str, CounterpartyCredit], times: np.ndarray) -> pd.DataFrame:
    """Survival, cumulative PD and marginal PD on ``times`` (which should start at 0) for each counterparty."""
    frames = []
    for credit in credits.values():
        curve = credit.hazard_curve()
        frames.append(pd.DataFrame({
            "counterparty_id": credit.counterparty_id,
            "time": times,
            "hazard": credit.hazard,
            "survival": curve.survival(times),
            "cumulative_pd": curve.cumulative_pd(times),
            "marginal_pd": np.concatenate([[0.0], curve.marginal_pd(times)]),
        }))
    return pd.concat(frames, ignore_index=True)
