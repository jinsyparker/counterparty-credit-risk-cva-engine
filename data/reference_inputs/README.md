# Reference inputs

Both files are synthetic, documented assumptions, not market data.

## `portfolio.csv`

Ten vanilla USD fixed-for-floating swaps in two netting sets, generated once by
`scripts/make_synthetic_portfolio.py` (fixed rate = par rate on the 2 Oct 2026 Treasury
curve ± a stated offset). Times are year fractions from the valuation date; `direction`
refers to the fixed leg from our side (`payer` = we pay fixed).

## `counterparties.csv`

| Column | Meaning |
|---|---|
| `cds_spread_bp` | **Assumed** counterparty spread scenario (CDS-equivalent), in bp. Not a market quote. |
| `recovery_rate` | Assumed recovery on default. |

Hazard rates use the credit-triangle approximation λ ≈ s / (1 − R). This is not a
calibration to a CDS curve: it ignores premium accrual, discounting inside the CDS legs
and the spread term structure. The CVA sensitivity tables show how results move with
both inputs.
