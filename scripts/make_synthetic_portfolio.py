"""Create the synthetic swap portfolio once, from the bootstrapped Treasury curve.

Each trade's fixed rate is its par rate on the snapshot curve plus a stated offset,
rounded to 0.01%. That gives a realistic mix of in- and out-of-the-money trades,
like a seasoned book. Once written, ``data/reference_inputs/portfolio.csv`` is the
source of truth; re-run this script only to rebuild the book from scratch.

Design of the book:
* NS_A (CPTY_A): mixed payers and receivers with offsetting maturities, so netting
  should remove a large share of the gross exposure. Includes a 1Y-into-5Y
  forward-starting swap.
* NS_B (CPTY_B): mostly payers (one-directional rate risk), so netting helps little.
"""

from __future__ import annotations

import pandas as pd

from src.config import CURVE_FILE, PORTFOLIO_FILE
from src.curves import load_treasury_curve
from src.instruments import PORTFOLIO_COLUMNS, Swap, portfolio_to_frame
from src.swap_pricer import par_rate

# trade_id, netting set, direction, notional, start, maturity, offset to par (bp)
TRADE_TERMS = [
    ("A1", "NS_A", "payer", 50_000_000, 0.0, 10.0, -50),
    ("A2", "NS_A", "receiver", 30_000_000, 0.0, 7.0, 25),
    ("A3", "NS_A", "payer", 25_000_000, 1.0, 6.0, 0),
    ("A4", "NS_A", "receiver", 40_000_000, 0.0, 10.0, -30),
    ("A5", "NS_A", "payer", 20_000_000, 0.0, 3.0, 40),
    ("A6", "NS_A", "receiver", 15_000_000, 0.0, 2.0, 60),
    ("B1", "NS_B", "payer", 30_000_000, 0.0, 5.0, -25),
    ("B2", "NS_B", "payer", 20_000_000, 0.0, 8.0, 20),
    ("B3", "NS_B", "payer", 25_000_000, 0.0, 4.0, -40),
    ("B4", "NS_B", "receiver", 10_000_000, 0.0, 3.0, 30),
]

COUNTERPARTY_OF = {"NS_A": "CPTY_A", "NS_B": "CPTY_B"}


def main() -> None:
    curve = load_treasury_curve(CURVE_FILE)
    swaps = []
    for trade_id, netting_set, direction, notional, start, maturity, offset_bp in TRADE_TERMS:
        template = Swap(trade_id, COUNTERPARTY_OF[netting_set], netting_set, notional, 0.0, direction, start, maturity)
        fixed_rate = round(par_rate(template, curve) + offset_bp / 10_000, 4)
        swaps.append(Swap(trade_id, COUNTERPARTY_OF[netting_set], netting_set, notional, fixed_rate, direction, start, maturity))

    table: pd.DataFrame = portfolio_to_frame(swaps)[PORTFOLIO_COLUMNS]
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(PORTFOLIO_FILE, index=False)
    print(f"Wrote {len(table)} trades -> {PORTFOLIO_FILE}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
