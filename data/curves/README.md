# Curve data

`ust_par_yields_2026-10-02.csv` is a snapshot of the **US Treasury daily par yield curve
(CMT) for 2 October 2026**, downloaded with `scripts/fetch_treasury_curve.py` from
<https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve>.

| Column | Meaning |
|---|---|
| `tenor_label` | Treasury column label (`1 Mo` ... `30 Yr`) |
| `tenor_years` | Tenor as a year fraction |
| `par_yield_pct` | Par yield, percent, semiannual bond-equivalent basis |

The snapshot is committed and is the source of truth for every run. Re-running the
fetch script writes a new dated file; point `src/config.py::CURVE_FILE` at it to switch.

**Simplification:** the engine is single-curve. These Treasury yields are used both to
discount and to project floating rates. A production USD swap engine would use a
SOFR/OIS curve instead (see the limitations section of the README).
