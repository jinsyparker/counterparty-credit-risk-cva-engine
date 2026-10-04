"""One-factor Hull-White short-rate model (closed-form pieces only).

Model
-----
    dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

Write r(t) = x(t) + phi(t), where x is a zero-mean Ornstein-Uhlenbeck process,

    dx(t) = -a x(t) dt + sigma dW(t),   x(0) = 0,
    phi(t) = f(0, t) + sigma^2 / (2 a^2) * (1 - e^{-a t})^2,

and theta(t) is whatever makes the model reproduce today's curve P(0, T) exactly.
Zero-coupon bond prices are then closed-form in x(t):

    B(t, T) = (1 - e^{-a (T - t)}) / a
    P(t, T) = P(0, T) / P(0, t)
              * exp(-B x(t) - 1/2 B^2 V_x(t) - B * sigma^2 / (2 a^2) * (1 - e^{-a t})^2)
    V_x(t)  = Var[x(t)] = sigma^2 / (2 a) * (1 - e^{-2 a t})

In this form the instantaneous forward f(0, t) cancels, so revaluation only needs
the discount curve and the simulated x(t). The stochastic discount factor along a
path is

    D(0, t) = exp(-int_0^t r ds) = P(0, t) * exp(-I(t) - 1/2 Var[I(t)]),   I(t) = int_0^t x ds,

which is a lognormal martingale correction: E[D(0, t)] = P(0, t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.curves import DiscountCurve


@dataclass(frozen=True)
class HullWhiteParams:
    """Hull-White parameters: mean-reversion speed a and normal volatility sigma."""

    mean_reversion: float
    volatility: float

    def __post_init__(self) -> None:
        if self.mean_reversion <= 0:
            raise ValueError("mean_reversion must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")


def bond_b(a: float, t: float, maturities: np.ndarray) -> np.ndarray:
    """B(t, T) = (1 - e^{-a (T - t)}) / a."""
    return -np.expm1(-a * (np.asarray(maturities, dtype=float) - t)) / a


def x_variance(params: HullWhiteParams, t: float | np.ndarray) -> float | np.ndarray:
    """Var[x(t)] = sigma^2 / (2a) * (1 - e^{-2 a t})."""
    a, sigma = params.mean_reversion, params.volatility
    return -sigma**2 / (2 * a) * np.expm1(-2 * a * np.asarray(t, dtype=float))


def phi_adjustment(params: HullWhiteParams, t: float | np.ndarray) -> float | np.ndarray:
    """phi(t) - f(0, t) = sigma^2 / (2 a^2) * (1 - e^{-a t})^2."""
    a, sigma = params.mean_reversion, params.volatility
    return sigma**2 / (2 * a**2) * np.expm1(-a * np.asarray(t, dtype=float)) ** 2


def integral_variance(params: HullWhiteParams, t: float | np.ndarray) -> float | np.ndarray:
    """Var[I(t)], I(t) = int_0^t x ds:  sigma^2 / a^2 * (t - 2 B(0, t) + (1 - e^{-2 a t}) / (2a))."""
    a, sigma = params.mean_reversion, params.volatility
    t = np.asarray(t, dtype=float)
    return sigma**2 / a**2 * (t + 2 * np.expm1(-a * t) / a - np.expm1(-2 * a * t) / (2 * a))


def zcb_price(curve: DiscountCurve, params: HullWhiteParams, t: float, maturities: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Hull-White P(t, T) for every path value x(t) and maturity T >= t.

    Returns an array of shape ``x.shape + maturities.shape`` (paths first, maturities
    on the last axis, as the swap pricer expects).
    """
    maturities = np.asarray(maturities, dtype=float)
    b = bond_b(params.mean_reversion, t, maturities)
    log_a = (
        np.asarray(curve.log_df(maturities))
        - curve.log_df(t)
        - 0.5 * b**2 * x_variance(params, t)
        - b * phi_adjustment(params, t)
    )
    return np.exp(log_a - np.multiply.outer(np.asarray(x, dtype=float), b))


def short_rate(curve: DiscountCurve, params: HullWhiteParams, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    """r(t) = x(t) + phi(t), with phi(t) = f(0, t) + sigma^2/(2a^2) (1 - e^{-at})^2."""
    return x + curve.inst_forward(t) + phi_adjustment(params, t)


def stochastic_discount(curve: DiscountCurve, params: HullWhiteParams, t: float | np.ndarray, integral_x: np.ndarray) -> np.ndarray:
    """D(0, t) = P(0, t) * exp(-I(t) - 1/2 Var[I(t)]), the pathwise bank-account discount factor."""
    return curve.df(t) * np.exp(-integral_x - 0.5 * integral_variance(params, t))


def ou_step_moments(params: HullWhiteParams, dt: float) -> tuple[float, float, float, float, float]:
    """Exact Gaussian transition of (x, I) over a step dt.

        x(t+dt)          = x(t) e^{-a dt}   + eps_x
        I(t+dt) - I(t)   = x(t) B(dt)       + eps_I

    Returns (e^{-a dt}, B(dt), Var[eps_x], Var[eps_I], Cov[eps_x, eps_I]) with

        Var[eps_x]        = sigma^2 / (2a) (1 - e^{-2 a dt})
        Var[eps_I]        = sigma^2 / a^2 (dt - 2 B(dt) + (1 - e^{-2 a dt}) / (2a))
        Cov[eps_x, eps_I] = sigma^2 / 2 * B(dt)^2
    """
    a, sigma = params.mean_reversion, params.volatility
    decay = np.exp(-a * dt)
    b = -np.expm1(-a * dt) / a
    var_x = float(x_variance(params, dt))
    var_i = float(integral_variance(params, dt))
    cov = 0.5 * sigma**2 * b**2
    return decay, b, var_x, var_i, cov
