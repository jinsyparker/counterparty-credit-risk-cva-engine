import numpy as np
import pytest

from src.config import PORTFOLIO_FILE
from src.instruments import Swap, group_by_netting_set, load_portfolio, make_schedule


def test_regular_schedule_covers_the_swap_life():
    sched = make_schedule(0.0, 5.0, 2)

    assert sched.pay_times.size == 10
    assert sched.pay_times[-1] == pytest.approx(5.0)
    assert sched.accrual_start[0] == 0.0
    np.testing.assert_allclose(sched.accruals, 0.5)


def test_forward_starting_schedule_begins_at_effective_date():
    sched = make_schedule(1.0, 6.0, 4)

    assert sched.accrual_start[0] == pytest.approx(1.0)
    assert sched.pay_times.size == 20
    assert sched.accruals.sum() == pytest.approx(5.0)


def test_odd_tenor_gets_short_first_period():
    sched = make_schedule(0.0, 2.25, 2)

    assert sched.accruals[0] == pytest.approx(0.25)
    np.testing.assert_allclose(sched.accruals[1:], 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [{"direction": "long"}, {"notional": -1.0}, {"start": 3.0, "maturity": 2.0}],
)
def test_invalid_swaps_raise(kwargs):
    terms = dict(trade_id="X", counterparty_id="C", netting_set_id="N", notional=1e6, fixed_rate=0.05,
                 direction="payer", start=0.0, maturity=5.0)
    terms.update(kwargs)
    with pytest.raises(ValueError):
        Swap(**terms)


def test_portfolio_file_loads_with_both_netting_sets_and_directions():
    portfolio = load_portfolio(PORTFOLIO_FILE)
    groups = group_by_netting_set(portfolio)

    assert len(portfolio) == 10
    assert set(groups) == {"NS_A", "NS_B"}
    assert sum(len(trades) for trades in groups.values()) == len(portfolio)
    assert {swap.direction for swap in portfolio} == {"payer", "receiver"}
    for netting_set, trades in groups.items():
        assert all(swap.netting_set_id == netting_set for swap in trades)
