"""Hull-White path simulation and pathwise revaluation of the swap portfolio.

Pipeline
--------
1. ``build_time_grid``: a regular grid (monthly by default) merged with every
   cash-flow, reset and effective date in the portfolio, so exposure is measured
   just after each payment and every floating fixing happens on a grid date.
2. ``simulate_paths``: exact Gaussian simulation of x(t) and I(t) = int x ds on that
   grid (no Euler discretization error, any step sizes).
3. ``revalue_portfolio``: at each grid date t, compute Hull-White P(t, T) for every
   future portfolio date, then price each trade with the same leg formulas as the
   time-0 pricer. Output: the MtM cube with shape (n_trades, n_paths, n_times).

Cash-flow timing: V(t) excludes cash flows paid at t. A floating coupon fixes at
its accrual start S_j as L_j = (1 / P(S_j, T_j) - 1) / tau_j on each path. While
S_j < t < T_j that stored fixing replaces the forward rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.curves import DiscountCurve
from src.instruments import Swap
from src.rate_models import HullWhiteParams, ou_step_moments, short_rate, stochastic_discount, zcb_price
from src.swap_pricer import fixed_leg_from_dfs, float_leg_from_dfs, forward_rates_from_dfs, swap_value_from_legs

TIME_DECIMALS = 10  # grid dates are rounded to this many decimals before merging
EPS = 1e-9


@dataclass(frozen=True)
class SimulationSettings:
    """Monte Carlo settings: number of paths, seed and regular grid step (years)."""

    n_paths: int = 10_000
    seed: int = 42
    time_step: float = 1 / 12


@dataclass(frozen=True)
class RatePaths:
    """Simulated Hull-White state on a time grid.

    ``x`` and ``integral_x`` have shape (n_paths, n_times); column 0 is t = 0.
    """

    times: np.ndarray
    x: np.ndarray
    integral_x: np.ndarray
    params: HullWhiteParams

    @property
    def n_paths(self) -> int:
        return self.x.shape[0]

    def short_rates(self, curve: DiscountCurve) -> np.ndarray:
        """r(t) on every path and grid date."""
        return short_rate(curve, self.params, self.times, self.x)

    def discount_factors(self, curve: DiscountCurve) -> np.ndarray:
        """Stochastic discount factor D(0, t) on every path and grid date."""
        return stochastic_discount(curve, self.params, self.times, self.integral_x)


def _round_times(times: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(times, dtype=float), TIME_DECIMALS)


def swap_dates(swap: Swap) -> np.ndarray:
    """Every date the swap needs on the grid: effective date, resets and payments."""
    fixed, flt = swap.fixed_schedule, swap.float_schedule
    return np.concatenate([[swap.start], fixed.pay_times, flt.accrual_start, flt.pay_times])


def build_time_grid(portfolio: list[Swap], step: float = 1 / 12, extra_dates: np.ndarray | None = None) -> np.ndarray:
    """Regular grid from 0 to the last maturity, merged with all portfolio dates and ``extra_dates``."""
    horizon = max(swap.maturity for swap in portfolio)
    n_steps = int(np.ceil(horizon / step - 1e-9))
    regular = np.minimum(np.arange(n_steps + 1) * step, horizon)
    pieces = [regular, *(swap_dates(swap) for swap in portfolio)]
    if extra_dates is not None:
        pieces.append(np.asarray(extra_dates, dtype=float))
    grid = np.unique(_round_times(np.concatenate(pieces)))
    return grid[(grid >= 0) & (grid <= horizon + EPS)]


def time_index(grid: np.ndarray, dates: float | np.ndarray) -> np.ndarray:
    """Positions of ``dates`` in ``grid``; raises if a date is not a grid point."""
    dates = _round_times(np.atleast_1d(dates))
    idx = np.searchsorted(grid, dates)
    idx = np.clip(idx, 0, grid.size - 1)
    if np.any(np.abs(grid[idx] - dates) > EPS):
        raise ValueError(f"dates not on the time grid: {dates[np.abs(grid[idx] - dates) > EPS]}")
    return idx


def simulate_paths(params: HullWhiteParams, grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> RatePaths:
    """Simulate x(t) and I(t) = int_0^t x ds exactly on ``grid`` (which must start at 0).

    Each step draws two independent standard normals per path and maps them through
    the Cholesky factor of the exact (eps_x, eps_I) covariance. All normals are drawn
    up front with shape (n_paths, n_steps, 2), so the same seed and grid give the same
    shocks for any parameter values (common random numbers for sensitivities).
    """
    grid = np.asarray(grid, dtype=float)
    if grid[0] != 0.0 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must start at 0 and be strictly increasing")
    shocks = rng.standard_normal((n_paths, grid.size - 1, 2))
    x = np.zeros((n_paths, grid.size))
    integral = np.zeros((n_paths, grid.size))
    for k in range(1, grid.size):
        decay, b, var_x, var_i, cov = ou_step_moments(params, grid[k] - grid[k - 1])
        l11 = np.sqrt(var_x)
        l21 = cov / l11 if l11 > 0 else 0.0
        l22 = np.sqrt(max(var_i - l21**2, 0.0))
        z1, z2 = shocks[:, k - 1, 0], shocks[:, k - 1, 1]
        x[:, k] = x[:, k - 1] * decay + l11 * z1
        integral[:, k] = integral[:, k - 1] + x[:, k - 1] * b + l21 * z1 + l22 * z2
    return RatePaths(times=grid, x=x, integral_x=integral, params=params)


def _step_covariance(params: HullWhiteParams, dt: float) -> tuple[float, float, np.ndarray]:
    """(e^{-a dt}, B(dt), 2x2 covariance of (eps_x, eps_I)) for one exact step."""
    decay, b, var_x, var_i, cov = ou_step_moments(params, dt)
    return decay, b, np.array([[var_x, cov], [cov, var_i]])


def _cholesky_2x2(cov: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a 2x2 PSD matrix, tolerant of a zero or near-singular matrix."""
    l11 = np.sqrt(max(cov[0, 0], 0.0))
    l21 = cov[1, 0] / l11 if l11 > 0 else 0.0
    l22 = np.sqrt(max(cov[1, 1] - l21**2, 0.0))
    return np.array([[l11, 0.0], [l21, l22]])


def refine_paths(paths: RatePaths, fine_grid: np.ndarray, rng: np.random.Generator) -> RatePaths:
    """Add dates to simulated paths without changing the existing ones (exact Gaussian bridge).

    Every original date keeps its simulated (x, I) exactly. Each new date s, taken in
    increasing order, is drawn from the exact conditional law of (x(s), I(s)) given the
    nearest known state on the left (t_L) and the next original state on the right (t_R):

        Y = (x_s, I_s) | left  ~ N(mu_Y, S1)                    (exact step t_L -> s)
        Z = (x_R, I_R)         = A Y + noise,  noise ~ N(0, S2)  (exact step s -> t_R)
        Y | Z  ~ N(mu_Y + K (Z - A mu_Y),  S1 - K A S1),   K = S1 A' (A S1 A' + S2)^{-1}

    with A = [[e^{-a dt2}, 0], [B(dt2), 1]]. Because (x, I) is Markov and Gaussian, the
    refined paths have exactly the joint law of a simulation on the fine grid, and any
    statistic on the original dates is unchanged (used for collateral look-back dates).
    """
    fine_grid = np.unique(_round_times(fine_grid))
    old = _round_times(paths.times)
    if not np.all(np.isin(old, fine_grid)):
        raise ValueError("fine_grid must contain every original date")
    if fine_grid[-1] > old[-1] + EPS:
        raise ValueError("fine_grid cannot extend beyond the last original date")

    n = paths.n_paths
    x = np.full((n, fine_grid.size), np.nan)
    integral = np.full((n, fine_grid.size), np.nan)
    pos_old = np.searchsorted(fine_grid, old)
    x[:, pos_old], integral[:, pos_old] = paths.x, paths.integral_x

    known = np.zeros(fine_grid.size, dtype=bool)
    known[pos_old] = True
    deterministic = paths.params.volatility == 0.0
    for j in np.flatnonzero(~known):
        left = np.flatnonzero(known[:j])[-1]
        right = j + np.flatnonzero(known[j + 1:])[0] + 1
        if deterministic:  # x is identically zero
            x[:, j] = integral[:, j] = 0.0
            known[j] = True
            continue
        d1, b1, s1 = _step_covariance(paths.params, fine_grid[j] - fine_grid[left])
        d2, b2, s2 = _step_covariance(paths.params, fine_grid[right] - fine_grid[j])
        a_mat = np.array([[d2, 0.0], [b2, 1.0]])
        gain = s1 @ a_mat.T @ np.linalg.inv(a_mat @ s1 @ a_mat.T + s2)
        cond_cov = s1 - gain @ a_mat @ s1
        mu_y = np.stack([d1 * x[:, left], integral[:, left] + b1 * x[:, left]], axis=1)
        z = np.stack([x[:, right], integral[:, right]], axis=1)
        mean = mu_y + (z - mu_y @ a_mat.T) @ gain.T
        draw = mean + rng.standard_normal((n, 2)) @ _cholesky_2x2(0.5 * (cond_cov + cond_cov.T)).T
        x[:, j], integral[:, j] = draw[:, 0], draw[:, 1]
        known[j] = True
    return RatePaths(times=fine_grid, x=x, integral_x=integral, params=paths.params)


@dataclass(frozen=True)
class _TradeLayout:
    """A swap's schedules as column indices into the portfolio's date vector."""

    swap: Swap
    fixed_pay: np.ndarray
    fixed_accruals: np.ndarray
    float_start: np.ndarray
    float_pay: np.ndarray
    float_accruals: np.ndarray
    float_start_col: np.ndarray
    float_pay_col: np.ndarray
    fixed_pay_col: np.ndarray


def _layout(swap: Swap, dates: np.ndarray) -> _TradeLayout:
    fixed, flt = swap.fixed_schedule, swap.float_schedule

    def col(t: np.ndarray) -> np.ndarray:
        return np.searchsorted(dates, _round_times(t))

    return _TradeLayout(
        swap=swap,
        fixed_pay=_round_times(fixed.pay_times),
        fixed_accruals=fixed.accruals,
        float_start=_round_times(flt.accrual_start),
        float_pay=_round_times(flt.pay_times),
        float_accruals=flt.accruals,
        float_start_col=col(flt.accrual_start),
        float_pay_col=col(flt.pay_times),
        fixed_pay_col=col(fixed.pay_times),
    )


def revalue_portfolio(portfolio: list[Swap], curve: DiscountCurve, paths: RatePaths) -> np.ndarray:
    """MtM cube V[trade, path, time] from our perspective, under the simulated Hull-White states.

    At each grid date t:
      * P(t, T) is computed once for every portfolio date T >= t (shape n_paths x n_dates);
      * coupons whose accrual starts at t record their fixing on each path;
      * each trade's remaining legs are priced with the time-0 leg formulas, using the
        stored fixing for a coupon that is already running.
    Trades are worth zero at and after their final payment date.
    """
    times = paths.times
    dates = np.unique(_round_times(np.concatenate([swap_dates(swap) for swap in portfolio])))
    layouts = [_layout(swap, dates) for swap in portfolio]
    fixings = [np.full((paths.n_paths, lay.float_pay.size), np.nan) for lay in layouts]
    cube = np.zeros((len(portfolio), paths.n_paths, times.size))
    discount = np.full((paths.n_paths, dates.size), np.nan)

    for k, t in enumerate(times):
        live = dates >= t - EPS
        if not live.any():
            break
        discount[:, live] = zcb_price(curve, paths.params, t, dates[live], paths.x[:, k])
        discount[:, ~live] = np.nan

        for i, lay in enumerate(layouts):
            swap = lay.swap
            if t >= swap.maturity - EPS:
                continue

            resets_now = np.abs(lay.float_start - t) <= EPS
            if resets_now.any():
                p_pay = discount[:, lay.float_pay_col[resets_now]]
                fixings[i][:, resets_now] = (1.0 / p_pay - 1.0) / lay.float_accruals[resets_now]

            fixed_rem = lay.fixed_pay > t + EPS
            pv_fixed = fixed_leg_from_dfs(
                swap.notional, swap.fixed_rate, lay.fixed_accruals[fixed_rem], discount[:, lay.fixed_pay_col[fixed_rem]]
            )

            float_rem = lay.float_pay > t + EPS
            started = lay.float_start[float_rem] < t - EPS
            p_pay = discount[:, lay.float_pay_col[float_rem]]
            p_start = np.where(started, 1.0, discount[:, lay.float_start_col[float_rem]])
            forwards = forward_rates_from_dfs(lay.float_accruals[float_rem], p_start, p_pay)
            if started.any():
                running = fixings[i][:, float_rem][:, started]
                if np.isnan(running).any():
                    raise ValueError(f"{swap.trade_id}: a reset date is missing from the simulation grid")
                forwards[:, started] = running
            pv_float = float_leg_from_dfs(swap.notional, lay.float_accruals[float_rem], forwards, p_pay)

            cube[i, :, k] = swap_value_from_legs(swap.sign, pv_float, pv_fixed)
    return cube


def netting_set_mtm(cube: np.ndarray, portfolio: list[Swap]) -> dict[str, np.ndarray]:
    """Sum the cube over the trades of each netting set: {netting_set_id: (n_paths, n_times)}."""
    ids = np.array([swap.netting_set_id for swap in portfolio])
    return {ns: cube[ids == ns].sum(axis=0) for ns in dict.fromkeys(ids)}
