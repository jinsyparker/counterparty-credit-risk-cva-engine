"""Download a US Treasury par-yield curve snapshot and save it as a CSV.

The committed snapshot in ``data/curves/`` is the source of truth for every run;
this script only exists so the snapshot can be re-created or refreshed.

Usage:
    python scripts/fetch_treasury_curve.py            # latest available date this year
    python scripts/fetch_treasury_curve.py 2026-10-02 # a specific date
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

# Treasury column label -> tenor in years
TENOR_COLUMNS = {
    "1 Mo": 1 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "curves"


def fetch_year(year: int) -> pd.DataFrame:
    """Return the Treasury daily par-yield table for one calendar year."""
    table = pd.read_csv(TREASURY_URL.format(year=year))
    table["Date"] = pd.to_datetime(table["Date"])
    return table.sort_values("Date")


def snapshot(table: pd.DataFrame, date: str | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Pick one curve date (latest by default) and return it in long format."""
    row = table.iloc[-1] if date is None else table.loc[table["Date"] == pd.Timestamp(date)].iloc[0]
    records = [
        {"tenor_label": label, "tenor_years": years, "par_yield_pct": float(row[label])}
        for label, years in TENOR_COLUMNS.items()
        if label in row.index and pd.notna(row[label])
    ]
    return row["Date"], pd.DataFrame(records)


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else None
    year = pd.Timestamp(date).year if date else pd.Timestamp.today().year
    curve_date, curve = snapshot(fetch_year(year), date)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ust_par_yields_{curve_date:%Y-%m-%d}.csv"
    curve.to_csv(out_path, index=False)
    print(f"Saved {len(curve)} tenors for {curve_date:%Y-%m-%d} -> {out_path}")
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
