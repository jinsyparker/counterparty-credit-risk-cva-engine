from dataclasses import replace

import numpy as np
import pytest

from src.config import CURVE_FILE, PORTFOLIO_FILE
from src.curves import DiscountCurve, load_treasury_curve
from src.instruments import Swap, load_portfolio
from src.swap_pricer import (
    cashflow_table,
    netting_set_summary,
    par_rate,
    price_portfolio,
    pv_fixed_leg,
    pv_float_leg,
    swap_mtm,
)

NOTIONAL = 10_000_000


@pytest.fixture(scope="module")
def curve():
    return load_treasury_curve(CURVE_FILE)


def make_swap(**overrides):
    terms = dict(trade_id="T", counterparty_id="C", netting_set_id="N", notional=NOTIONAL, fixed_rate=0.05,
                 direction="payer", start=0.0, maturity=5.0)
    terms.update(overrides)
    return Swap(**terms)


@pytest.mark.parametrize("start, maturity", [(0.0, 2.0), (0.0, 5.0), (0.0, 10.0), (1.0, 6.0), (2.0, 4.75)])
@pytest.mark.parametrize("direction", ["payer", "receiver"])
def test_par_swap_has_zero_value(curve, start, maturity, direction):
    swap = make_swap(start=start, maturity=maturity, direction=direction)
    at_par = replace(swap, fixed_rate=par_rate(swap, curve))

    assert swap_mtm(at_par, curve) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("fixed_rate", [0.03, 0.05, 0.07])
def test_payer_receiver_opposite(curve, fixed_rate):
    payer = make_swap(fixed_rate=fixed_rate, direction="payer")
    receiver = replace(payer, direction="receiver")

    assert swap_mtm(payer, curve) == pytest.approx(-swap_mtm(receiver, curve), abs=1e-8)
    assert swap_mtm(payer, curve) != 0.0


def test_float_leg_telescopes_to_discount_factor_difference(curve):
    swap = make_swap(start=1.0, maturity=6.0)

    expected = NOTIONAL * (curve.df(1.0) - curve.df(6.0))
    assert pv_float_leg(swap, curve) == pytest.approx(expected, rel=1e-12)


def test_fixed_leg_matches_hand_calculation():
    # Flat 4% continuous curve, 2Y annual fixed leg at 5%: N * 0.05 * (e^-0.04 + e^-0.08)
    curve = DiscountCurve(np.array([1.0, 2.0]), np.array([0.04, 0.04]))
    swap = make_swap(maturity=2.0, fixed_freq=1)

    expected = NOTIONAL * 0.05 * (np.exp(-0.04) + np.exp(-0.08))
    assert pv_fixed_leg(swap, curve) == pytest.approx(expected, rel=1e-12)


def test_par_rate_on_flat_curve_matches_closed_form():
    # Flat continuous rate r, semiannual fixed leg: K_par = 2 (e^{r/2} - 1) for any maturity
    r = 0.045
    curve = DiscountCurve(np.array([1.0, 30.0]), np.array([r, r]))

    for maturity in (1.0, 3.5, 10.0):
        assert par_rate(make_swap(maturity=maturity), curve) == pytest.approx(2 * (np.exp(r / 2) - 1), abs=1e-13)


def test_payer_gains_when_fixed_rate_is_below_par(curve):
    swap = make_swap()
    k_par = par_rate(swap, curve)

    assert swap_mtm(replace(swap, fixed_rate=k_par - 0.005), curve) > 0
    assert swap_mtm(replace(swap, fixed_rate=k_par + 0.005), curve) < 0


def test_cashflow_table_pv_sums_to_mtm(curve):
    swap = make_swap(fixed_rate=0.045, start=1.0, maturity=6.0)
    table = cashflow_table(swap, curve)

    assert table["pv"].sum() == pytest.approx(swap_mtm(swap, curve), rel=1e-10)
    assert set(table["leg"]) == {"fixed", "float"}


def test_portfolio_aggregation_is_generic(curve):
    portfolio = load_portfolio(PORTFOLIO_FILE)
    valuation = price_portfolio(portfolio, curve)
    summary = netting_set_summary(valuation).set_index("netting_set_id")

    for netting_set, trades in valuation.groupby("netting_set_id"):
        assert summary.loc[netting_set, "net_mtm"] == pytest.approx(trades["mtm"].sum(), rel=1e-12)
        assert summary.loc[netting_set, "net_mtm"] <= summary.loc[netting_set, "gross_positive_mtm"] + 1e-9

    flipped = [replace(s, direction="receiver" if s.direction == "payer" else "payer") for s in portfolio]
    assert price_portfolio(flipped, curve)["mtm"].sum() == pytest.approx(-valuation["mtm"].sum(), rel=1e-12)
