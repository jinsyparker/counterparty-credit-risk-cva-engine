import numpy as np
import pytest

from src.collateral import TWO_WEEKS, CsaTerms, collateral_balance, collateralized_exposure, lookback_dates
from src.config import COUNTERPARTIES_FILE, CURVE_FILE, HW_PARAMS, PORTFOLIO_FILE
from src.credit import load_counterparties
from src.curves import load_treasury_curve
from src.cva import estimate_cva
from src.exposure import expected_exposure, expected_positive_exposure, positive_exposure
from src.instruments import load_portfolio
from src.simulation import build_time_grid, netting_set_mtm, refine_paths, revalue_portfolio, simulate_paths, time_index

MPORS = (7 / 365, TWO_WEEKS, 28 / 365)


@pytest.fixture(scope="module")
def setup():
    curve = load_treasury_curve(CURVE_FILE)
    portfolio = load_portfolio(PORTFOLIO_FILE)
    grid = build_time_grid(portfolio)
    base = simulate_paths(HW_PARAMS, grid, 3_000, np.random.default_rng(17))
    fine_grid = np.unique(np.concatenate([grid, *(lookback_dates(grid, m) for m in MPORS)]))
    fine = refine_paths(base, fine_grid, np.random.default_rng(18))
    values = netting_set_mtm(revalue_portfolio(portfolio, curve, fine), portfolio)
    return {
        "grid": grid, "fine_grid": fine_grid, "values": values,
        "now": {ns: v[:, time_index(fine_grid, grid)] for ns, v in values.items()},
        "discount": base.discount_factors(curve), "credits": load_counterparties(COUNTERPARTIES_FILE),
    }


def lagged(setup, ns, mpor):
    return setup["values"][ns][:, time_index(setup["fine_grid"], lookback_dates(setup["grid"], mpor))]


def cva(setup, ns, exposure):
    credit = setup["credits"][ns]
    return estimate_cva(exposure, setup["discount"], credit.hazard_curve().marginal_pd(setup["grid"]), credit.lgd).value


def test_infinite_threshold_equals_uncollateralized(setup):
    for ns, now in setup["now"].items():
        exposure = collateralized_exposure(now, lagged(setup, ns, TWO_WEEKS), CsaTerms("none"))
        np.testing.assert_array_equal(exposure, positive_exposure(now))


def test_collateral_never_increases_exposure(setup):
    terms = CsaTerms("zero", threshold=0.0)
    for ns, now in setup["now"].items():
        assert np.all(collateralized_exposure(now, lagged(setup, ns, TWO_WEEKS), terms) <= positive_exposure(now) + 1e-9)


def test_tighter_threshold_reduces_exposure_and_cva(setup):
    # Lower threshold -> more collateral -> weakly lower exposure on every path, lower CVA (spec section 11)
    thresholds = [np.inf, 5e6, 2e6, 5e5, 1e5, 0.0]
    for ns, now in setup["now"].items():
        lag = lagged(setup, ns, TWO_WEEKS)
        exposures = [collateralized_exposure(now, lag, CsaTerms("h", threshold=h)) for h in thresholds]
        for looser, tighter in zip(exposures, exposures[1:]):
            assert np.all(tighter <= looser + 1e-9)
        cvas = [cva(setup, ns, e) for e in exposures]
        assert np.all(np.diff(cvas) <= 1e-9)
        assert cvas[-1] < 0.25 * cvas[0]


def test_larger_minimum_transfer_increases_exposure(setup):
    for ns, now in setup["now"].items():
        lag = lagged(setup, ns, TWO_WEEKS)
        small = collateralized_exposure(now, lag, CsaTerms("m", threshold=0.0, minimum_transfer=1e5))
        large = collateralized_exposure(now, lag, CsaTerms("m", threshold=0.0, minimum_transfer=5e5))
        assert np.all(large >= small - 1e-9)


def test_zero_threshold_and_zero_mpor_remove_all_exposure(setup):
    terms = CsaTerms("perfect", threshold=0.0, mpor=0.0)
    for now in setup["now"].values():
        np.testing.assert_array_equal(collateralized_exposure(now, now, terms), 0.0)


def test_longer_margin_period_means_more_residual_exposure(setup):
    # Under a zero threshold the residual exposure comes from value moves inside the MPoR
    for ns, now in setup["now"].items():
        epes = [expected_positive_exposure(setup["grid"], expected_exposure(
                collateralized_exposure(now, lagged(setup, ns, m), CsaTerms("z", threshold=0.0, mpor=m))))
                for m in MPORS]
        assert epes[0] < epes[1] < epes[2]


def test_collateral_is_one_way():
    lagged_values = np.array([[-3e6, -1.0, 0.0, 1e6, 4e6]])
    terms = CsaTerms("h", threshold=1e6, minimum_transfer=5e5)

    np.testing.assert_allclose(collateral_balance(lagged_values, terms), [[0.0, 0.0, 0.0, 0.0, 2.5e6]])


def test_lookback_before_inception_uses_time_zero():
    np.testing.assert_allclose(lookback_dates(np.array([0.0, 0.02, 1.0]), TWO_WEEKS), [0.0, 0.0, 1.0 - TWO_WEEKS])


@pytest.mark.parametrize("kwargs", [{"threshold": -1.0}, {"minimum_transfer": -1.0}, {"mpor": -0.01}])
def test_invalid_csa_terms_raise(kwargs):
    with pytest.raises(ValueError):
        CsaTerms("bad", **kwargs)
