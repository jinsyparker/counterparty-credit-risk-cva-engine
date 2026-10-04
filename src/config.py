"""Project paths and base-case run settings.

Every assumption a run depends on lives here (or in the CSVs under ``data/``),
so a result can always be traced back to a single documented input.
"""

from __future__ import annotations

from pathlib import Path

from src.collateral import CsaTerms
from src.rate_models import HullWhiteParams
from src.simulation import SimulationSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CURVE_FILE = DATA_DIR / "curves" / "ust_par_yields_2026-10-02.csv"
VALUATION_DATE = "2026-10-02"
PORTFOLIO_FILE = DATA_DIR / "reference_inputs" / "portfolio.csv"
COUNTERPARTIES_FILE = DATA_DIR / "reference_inputs" / "counterparties.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- Hull-White simulation ---------------------------------------------------------
# a = 3%/yr mean reversion and sigma = 1%/sqrt(yr), the absolute (normal) diffusion
# volatility of the short rate. sigma is a model parameter, not a quoted swaption
# normal (Bachelier) vol; both values are assumed, not calibrated to swaptions.
# Phase F reports sensitivities to both.
HW_PARAMS = HullWhiteParams(mean_reversion=0.03, volatility=0.01)
BASE_SIMULATION = SimulationSettings(n_paths=10_000, seed=42, time_step=1 / 12)
QUICK_SIMULATION = SimulationSettings(n_paths=2_000, seed=42, time_step=1 / 12)

# --- Exposure ------------------------------------------------------------------------
PFE_QUANTILE = 0.95
DISTRIBUTION_TIMES = (1.0, 3.0, 5.0)  # dates for the exposure-distribution figure

# --- Credit & CVA --------------------------------------------------------------------
# Assumed counterparty spread scenarios (CDS-equivalent) and recoveries live in
# data/reference_inputs/counterparties.csv. Hazards use the credit-triangle
# approximation lambda ~ s / (1 - R), not a CDS curve calibration. These grids drive the
# credit-only CVA sensitivity table, which reuses the same simulated exposures.
SPREAD_GRID_BP = (50, 100, 150, 200, 300, 500)
RECOVERY_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)

# --- Collateral ----------------------------------------------------------------------
# Simplified one-way CSA (the counterparty posts, we never do) with a two-week margin
# period of risk. Look-back dates t - MPoR are added to the Phase D paths by an exact
# Gaussian bridge drawn from rng(seed + BRIDGE_SEED_OFFSET), so every original date and
# the uncollateralized CVA are unchanged.
CSA_SCENARIOS = (
    CsaTerms("Uncollateralized"),
    CsaTerms("Threshold $2m, MTA $250k", threshold=2_000_000, minimum_transfer=250_000),
    CsaTerms("Threshold $500k, MTA $100k", threshold=500_000, minimum_transfer=100_000),
    CsaTerms("Zero threshold, no MTA", threshold=0.0, minimum_transfer=0.0),
)
MPOR_GRID_DAYS = (7, 14, 28)  # margin-period-of-risk sensitivity under the zero-threshold CSA
BRIDGE_SEED_OFFSET = 1

# --- Validation: convergence and market sensitivities ------------------------------
# Convergence pools independent chunks of BASE/QUICK n_paths. Chunk k uses seed
# seed + CHUNK_SEED_STRIDE * k, so chunk 0 is exactly the base case. Market
# sensitivities re-simulate with the base seed (common random numbers).
CONVERGENCE_PATHS = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
CHUNK_SEED_STRIDE = 1_000
SIGMA_GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015)
MEAN_REVERSION_GRID = (0.01, 0.03, 0.05, 0.10)
MATURITY_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)  # tenor multipliers; trades re-struck at par + original offset
NOTIONAL_SCALES = (0.5, 1.0, 1.5, 2.0)
