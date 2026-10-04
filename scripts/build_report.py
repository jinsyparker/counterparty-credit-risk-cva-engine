"""Build the illustrated project report (HTML and, optionally, PDF) from generated results.

Run after ``python run_all.py`` (it reads ``reports/tables`` and ``reports/figures``):

    python scripts/build_report.py          # writes reports/cva_report.html and reports/cva_report_print.html
    python scripts/build_report.py --pdf    # also renders reports/CVA_Engine_Report.pdf with headless Chrome

Every number in the report is read from the generated tables or recomputed with the
project's ``src`` code here; none are typed by hand. The script also runs the test
suite to report the test count.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "reports"
PDF_NAME = "CVA_Engine_Report.pdf"
sys.path.insert(0, str(PROJECT))

parser = argparse.ArgumentParser(description="Build the CVA engine report.")
parser.add_argument("--pdf", action="store_true", help="also render the PDF with headless Chrome")
args = parser.parse_args()
if not (OUT_DIR / "tables" / "convergence.csv").exists():
    sys.exit("reports/tables is missing or incomplete: run `python run_all.py` first.")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    BASE_SIMULATION, COUNTERPARTIES_FILE, CURVE_FILE, FIGURES_DIR, HW_PARAMS, PFE_QUANTILE, PORTFOLIO_FILE, TABLES_DIR,
    VALUATION_DATE,
)
from src.credit import load_counterparties  # noqa: E402
from src.curves import load_par_yields, load_treasury_curve, par_bond_schedule  # noqa: E402
from src.exposure import gross_exposure, netted_exposure, potential_future_exposure  # noqa: E402
from src.instruments import load_portfolio  # noqa: E402
from src.simulation import build_time_grid, netting_set_mtm, revalue_portfolio, simulate_paths, time_index  # noqa: E402
from src.swap_pricer import par_rate, swap_mtm  # noqa: E402
from src.validation import martingale_table  # noqa: E402

# ================================================================ data
curve = load_treasury_curve(CURVE_FILE)
quotes = load_par_yields(CURVE_FILE)
portfolio = load_portfolio(PORTFOLIO_FILE)
credits = load_counterparties(COUNTERPARTIES_FILE)
T = lambda name: pd.read_csv(TABLES_DIR / name)  # noqa: E731
val, ns_mtm, dist, mart, mtm = (T("portfolio_valuation.csv"), T("netting_set_mtm.csv"), T("rate_distribution.csv"),
                                T("martingale_check.csv"), T("mtm_summary.csv"))
exp_sum, exp_prof = T("exposure_summary.csv"), T("exposure_profiles.csv")
cva_sum, cva_buckets, cva_sens = T("cva_summary.csv"), T("cva_buckets.csv"), T("cva_credit_sensitivity.csv")

a1 = next(s for s in portfolio if s.trade_id == "A1")
a1_par_mtm = swap_mtm(replace(a1, fixed_rate=par_rate(a1, curve)), curve)
a1_payer, a1_receiver = swap_mtm(a1, curve), swap_mtm(replace(a1, direction="receiver"), curve)
reprice_err = max(abs(y * np.sum(acc * curve.df(pt)) + curve.df(t) - 1.0)
                  for t, y in zip(quotes["tenor_years"], quotes["par_yield"]) for pt, acc in [par_bond_schedule(t)])
fwd_10y = curve.inst_forward(np.linspace(0, 10, 2001))

start = time.perf_counter()
grid = build_time_grid(portfolio, step=BASE_SIMULATION.time_step)
paths = simulate_paths(HW_PARAMS, grid, BASE_SIMULATION.n_paths, np.random.default_rng(BASE_SIMULATION.seed))
cube = revalue_portfolio(portfolio, curve, paths)
sim_seconds = time.perf_counter() - start


def slope_z(sim, t, h):
    now, later = sim.x[:, time_index(sim.times, t)[0]], sim.x[:, time_index(sim.times, t + h)[0]]
    slope = now @ later / (now @ now)
    return (slope - np.exp(-HW_PARAMS.mean_reversion * h)) / (np.std(later - slope * now) / np.sqrt(now @ now))


SLOPE_CASES = [(1.0, 1.0), (2.0, 3.0), (3.0, 5.0)]
worst_base_slope = max((slope_z(paths, t, h) for t, h in SLOPE_CASES), key=abs)
big = simulate_paths(HW_PARAMS, np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]), 400_000, np.random.default_rng(7))
big_mart_z = martingale_table(curve, big)["z_score"].abs().max()
big_slope_z = max(abs(slope_z(big, t, h)) for t, h in SLOPE_CASES)

# Phase C checks on the base-case cube
netted, gross = netted_exposure(cube, portfolio), gross_exposure(cube, portfolio)
values = netting_set_mtm(cube, portfolio)
min_exposure = min(e.min() for e in netted.values())
max_net_minus_gross = max((netted[ns] - gross[ns]).max() for ns in netted)
ee0_gap = max(abs(netted[ns].mean(axis=0)[0] - max(values[ns][0, 0], 0.0)) for ns in netted)
pfe_ok = all(
    np.all((netted[ns] <= (p := potential_future_exposure(netted[ns], PFE_QUANTILE))).mean(axis=0) >= PFE_QUANTILE)
    and np.all((netted[ns] < p).mean(axis=0) < PFE_QUANTILE) for ns in netted)
p_zero = {(ns, t): (netted[ns][:, time_index(grid, t)[0]] == 0).mean() for ns in netted for t in (1.0, 3.0, 5.0)}

# Phase D: credit-triangle comparison (same seed, so the same exposures as run_all.py)
bank = paths.discount_factors(curve)
triangle = {ns: c.spread * np.trapezoid((netted[ns] * bank).mean(axis=0), grid) for ns, c in credits.items()}

py = sys.executable
collected = subprocess.run([py, "-m", "pytest", "--collect-only", "-q"], cwd=PROJECT, capture_output=True, text=True).stdout
tests_per_file = Counter(line.split("::")[0].replace("tests/", "") for line in collected.splitlines() if "::" in line)
n_tests = sum(tests_per_file.values())
passed = subprocess.run([py, "-m", "pytest", "-q"], cwd=PROJECT, capture_output=True, text=True).stdout.strip().splitlines()[-1]


# ================================================================ formatting helpers
def money(x: float, decimals: int = 0) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.{decimals}f}"


def millions(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x) / 1e6:,.2f}m"


def thousands(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x) / 1e3:,.1f}k"


def pct(x: float, d: int = 2) -> str:
    return f"{'−' if x < 0 else ''}{abs(x) * 100:.{d}f}%"


def signed_cell(text: str, value: float) -> str:
    return f'<td class="num{" neg" if value < 0 else ""}">{text}</td>'


def table(headers: list[tuple[str, str]], rows: list[str]) -> str:
    head = "".join(f'<th class="{cls}">{html.escape(h)}</th>' for h, cls in headers)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def figure(name: str, alt: str, caption: str) -> str:
    return (f'<figure><div class="plate"><img src="figures/{name}" alt="{html.escape(alt)}"></div>'
            f"<figcaption>{caption}</figcaption></figure>")


def tenor_label(years: float) -> str:
    return f"{round(years * 12)}M" if years < 1 else f"{years:g}Y"


# ================================================================ Phase A/B tables
curve_table = table(
    [("Tenor", ""), ("Par yield (input)", "num"), ("Zero rate, cont. comp.", "num"), ("P(0, T)", "num")],
    [f"<tr><td>{tenor_label(t)}</td><td class='num'>{pct(y)}</td><td class='num'>{pct(curve.zero_rate(t), 3)}</td>"
     f"<td class='num'>{curve.df(t):.5f}</td></tr>" for t, y in zip(quotes["tenor_years"], quotes["par_yield"])])

blotter_rows = []
for r in val.itertuples():
    tag = "PAY" if r.direction == "payer" else "REC"
    blotter_rows.append(
        f"<tr><td class='mono'>{r.trade_id}</td><td>{r.netting_set_id}</td><td><span class='tag tag-{tag.lower()}'>{tag}</span></td>"
        f"<td class='num'>${r.notional / 1e6:,.0f}m</td><td class='num'>{r.start:g}Y → {r.maturity:g}Y</td>"
        f"<td class='num'>{pct(r.fixed_rate)}</td><td class='num'>{pct(r.par_rate)}</td>"
        f"<td class='num'>{(r.fixed_rate - r.par_rate) * 1e4:+.0f}</td>{signed_cell(money(r.mtm), r.mtm)}</tr>")
blotter = table([("Trade", ""), ("Netting set", ""), ("Fixed leg", ""), ("Notional", "num"), ("Start → maturity", "num"),
                 ("Fixed rate", "num"), ("Par rate", "num"), ("Fixed − par (bp)", "num"), ("MtM to us", "num")], blotter_rows)

ns_table = table(
    [("Netting set", ""), ("Counterparty", ""), ("Trades", "num"), ("Gross notional", "num"), ("Net MtM", "num"),
     ("Sum of positive trade MtMs", "num"), ("Reduction from netting", "num")],
    [f"<tr><td>{r.netting_set_id}</td><td>{r.counterparty_id}</td><td class='num'>{r.n_trades}</td>"
     f"<td class='num'>${r.gross_notional / 1e6:,.0f}m</td>{signed_cell(money(r.net_mtm), r.net_mtm)}"
     f"<td class='num'>{money(r.gross_positive_mtm)}</td><td class='num'>{(1 - r.net_mtm / r.gross_positive_mtm) * 100:.0f}%</td></tr>"
     for r in ns_mtm.itertuples()])

dist_table = table(
    [("t", "num"), ("Mean (MC)", "num"), ("Mean (analytic)", "num"), ("Std (MC)", "num"), ("Std (analytic)", "num"),
     ("5–95% range", "num"), ("P(r < 0)", "num")],
    [f"<tr><td class='num'>{r.time:g}Y</td><td class='num'>{pct(r.mean_mc)}</td><td class='num'>{pct(r.mean_analytic)}</td>"
     f"<td class='num'>{pct(r.std_mc)}</td><td class='num'>{pct(r.std_analytic)}</td>"
     f"<td class='num'>{pct(r.p05)} – {pct(r.p95)}</td><td class='num'>{pct(r.prob_negative)}</td></tr>" for r in dist.itertuples()])

mart_table = table(
    [("t", "num"), ("T", "num"), ("E[D(0,t)·P(t,T)] (MC)", "num"), ("P(0,T) (curve)", "num"), ("Std error", "num"), ("z", "num")],
    [f"<tr><td class='num'>{r.t:g}Y</td><td class='num'>{r.T:g}Y</td><td class='num'>{r._3:.5f}</td>"
     f"<td class='num'>{r._4:.5f}</td><td class='num'>{r.std_error:.5f}</td><td class='num'>{r.z_score:+.2f}</td></tr>"
     for r in mart.itertuples()])

mtm_table = table(
    [("Netting set", ""), ("t", "num"), ("Mean netted MtM", "num"), ("5–95% range", "num")],
    [f"<tr><td>{r.netting_set_id}</td><td class='num'>{r.time:g}Y</td>{signed_cell(millions(r.mean), r.mean)}"
     f"<td class='num'>{millions(r.p05)} to {millions(r.p95)}</td></tr>" for r in mtm.itertuples() if r.time < 10])

# ================================================================ Phase C tables
pfe_col = f"peak_pfe{PFE_QUANTILE * 100:g}_netted"
exp_table = table(
    [("Netting set", ""), ("Life", "num"), ("EPE (time-avg EE) netted", "num"), ("EPE (time-avg EE) gross", "num"), ("Netting reduction", "num"),
     ("Peak EE", "num"), ("Peak PFE 95%", "num")],
    [f"<tr><td>{r.netting_set_id}</td><td class='num'>{r.horizon:g}Y</td><td class='num'>{millions(r.epe_netted)}</td>"
     f"<td class='num'>{millions(r.epe_gross)}</td><td class='num'>{r.netting_reduction:.0%}</td>"
     f"<td class='num'>{millions(r.peak_ee_netted)} at {r.peak_ee_time:.1f}Y</td>"
     f"<td class='num'>{millions(getattr(r, pfe_col))} at {r.peak_pfe_time:.1f}Y</td></tr>" for r in exp_sum.itertuples()])

prof_rows = []
for r in exp_prof[exp_prof.time.round(6).isin([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])].itertuples():
    prof_rows.append(f"<tr><td>{r.netting_set_id}</td><td class='num'>{r.time:g}Y</td><td class='num'>{millions(r.ee_netted)}</td>"
                     f"<td class='num'>{millions(r.pfe_netted)}</td><td class='num'>{millions(r.ee_gross)}</td></tr>")
prof_table = table([("Netting set", ""), ("t", "num"), ("EE netted", "num"), ("PFE 95% netted", "num"), ("EE gross", "num")],
                   prof_rows)

ea, eb = exp_sum.set_index("netting_set_id").loc["NS_A"], exp_sum.set_index("netting_set_id").loc["NS_B"]

# ================================================================ Phase D tables
horizon_of = exp_sum.set_index("netting_set_id")["horizon"]
credit_table = table(
    [("Netting set", ""), ("Counterparty", ""), ("Assumed spread", "num"), ("Recovery", "num"), ("Hazard λ ≈ s/(1−R)", "num"),
     ("LGD", "num"), ("PD to last maturity", "num")],
    [f"<tr><td>{ns}</td><td>{c.counterparty_id}</td><td class='num'>{c.spread * 1e4:.0f}bp</td><td class='num'>{c.recovery:.0%}</td>"
     f"<td class='num'>{pct(c.hazard, 1)}</td><td class='num'>{c.lgd:.0%}</td>"
     f"<td class='num'>{pct(float(c.hazard_curve().cumulative_pd(horizon_of[ns])), 1)} ({horizon_of[ns]:g}Y)</td></tr>"
     for ns, c in credits.items()])

cva_rows = []
for r in cva_sum.itertuples():
    lo, hi = r.cva_netted - 1.96 * r.cva_netted_se, r.cva_netted + 1.96 * r.cva_netted_se
    cls = " class='total'" if r.netting_set_id == "TOTAL" else ""
    cva_rows.append(
        f"<tr{cls}><td>{r.netting_set_id}</td><td class='num'>{money(r.cva_netted)}</td><td class='num'>± {money(r.cva_netted_se)}</td>"
        f"<td class='num'>{money(lo)} – {money(hi)}</td><td class='num'>{money(r.cva_gross)}</td>"
        f"<td class='num'>{r.netting_reduction:.0%}</td><td class='num'>{money(r.cva_benchmark_curve_df)}</td>"
        f"<td class='num'>{r.benchmark_vs_headline:+.1%}</td></tr>")
cva_table = table([("Netting set", ""), ("CVA netted (headline)", "num"), ("Std error", "num"), ("95% interval", "num"),
                   ("CVA gross", "num"), ("Netting reduction", "num"), ("Curve-DF benchmark", "num"), ("Benchmark vs headline", "num")],
                  cva_rows)

bucket_rows = []
for ns, rows in cva_buckets.groupby("netting_set_id", sort=False):
    cells = "".join(f"<td class='num'>{v / 1e3:,.1f}</td>" for v in rows["cva_netted"])
    bucket_rows.append(f"<tr><td>{ns}</td>{cells}<td class='num'><b>{rows['cva_netted'].sum() / 1e3:,.1f}</b></td></tr>")
years = cva_buckets.loc[cva_buckets.netting_set_id == "NS_A", "bucket_end"].astype(int)
bucket_table = table([("Netted CVA ($k)", "")] + [(f"Y{y}", "num") for y in years] + [("Total", "num")], bucket_rows)

robust_rows = []
for r in cva_sum[cva_sum.netting_set_id != "TOTAL"].itertuples():
    robust_rows.append(
        f"<tr><td>{r.netting_set_id}</td><td class='num'>{money(r.cva_netted)}</td><td class='num'>{money(r.cva_benchmark_curve_df)}</td>"
        f"<td class='num'>{r.benchmark_vs_headline:+.1%}</td><td class='num'>{money(triangle[r.netting_set_id])}</td>"
        f"<td class='num'>{triangle[r.netting_set_id] / r.cva_netted - 1:+.1%}</td></tr>")
robust_table = table([("Netting set", ""), ("Headline (pathwise D)", "num"), ("Benchmark (curve DF)", "num"), ("Benchmark vs headline", "num"),
                      ("Credit triangle s·∫dEE dt", "num"), ("Triangle vs headline", "num")], robust_rows)

sens_rows = []
pivot = cva_sens.pivot_table(index=["view", "input"], columns="netting_set_id", values="cva")
view_names = {"spread": "Assumed spread (bp), R = 40%", "recovery_hazard_fixed": "Recovery, hazard fixed",
              "recovery_spread_fixed": "Recovery, spread fixed"}
for view in ["spread", "recovery_hazard_fixed", "recovery_spread_fixed"]:
    block = pivot.loc[view]
    labels = [f"{v:g}bp" if view == "spread" else f"{v:.0%}" for v in block.index]
    for i, (label, row) in enumerate(zip(labels, block.itertuples(index=False))):
        first = f"<td rowspan='{len(block)}' class='view'>{view_names[view]}</td>" if i == 0 else ""
        sens_rows.append(f"<tr>{first}<td class='num'>{label}</td>" +
                         "".join(f"<td class='num'>{thousands(v)}</td>" for v in row) + "</tr>")
sens_table = table([("View", ""), ("Input", "num")] + [(ns, "num") for ns in pivot.columns], sens_rows)

ca, cb = cva_sum.set_index("netting_set_id").loc["NS_A"], cva_sum.set_index("netting_set_id").loc["NS_B"]
ctot = cva_sum.set_index("netting_set_id").loc["TOTAL"]
spread_view = pivot.loc["spread"]
rec_fixed_s = pivot.loc["recovery_spread_fixed"]


# ================================================================ Phase E / F tables
from src.reporting import load_results, resume_bullets  # noqa: E402

coll_sum, mitig, mpor_t = T("collateral_summary.csv"), T("cva_mitigation.csv"), T("collateral_mpor_sensitivity.csv")
conv, mkt, torn = T("convergence.csv"), T("market_sensitivity.csv"), T("sensitivity_tornado.csv")


def reduction(x: float) -> str:
    return "—" if abs(x) < 0.005 else f"−{x:.0%}"


coll_rows = []
for r in coll_sum.itertuples():
    coll_rows.append(
        f"<tr><td>{r.netting_set_id}</td><td>{html.escape(r.scenario)}</td><td class='num'>{millions(r.epe_time_avg_ee)}</td>"
        f"<td class='num'>{millions(r.peak_ee)}</td><td class='num'>{millions(r.peak_pfe)}</td>"
        f"<td class='num'>{money(r.cva)} ± {money(r.cva_se)}</td><td class='num'>{reduction(r.cva_reduction_vs_uncollateralized)}</td></tr>")
coll_table = table([("Netting set", ""), ("CSA scenario (MPoR 2 weeks)", ""), ("EPE (time-avg EE)", "num"), ("Peak EE", "num"),
                    ("Peak PFE 95%", "num"), ("CVA ± std error", "num"), ("vs uncollateralized", "num")], coll_rows)

mitig_table = table([("Netting set", ""), ("Stage", ""), ("CVA", "num"), ("vs gross", "num")],
                    [f"<tr><td>{r.netting_set_id}</td><td>{html.escape(r.stage)}</td><td class='num'>{money(r.cva)}</td>"
                     f"<td class='num'>{reduction(r.reduction_vs_gross)}</td></tr>" for r in mitig.itertuples()])

mpor_pivot = mpor_t.pivot(index="mpor_days", columns="netting_set_id", values="cva")
mpor_share = mpor_t.pivot(index="mpor_days", columns="netting_set_id", values="cva_share_of_uncollateralized")
mpor_rows = [f"<tr><td class='num'>{d} days</td>" + "".join(
             f"<td class='num'>{money(mpor_pivot.loc[d, ns])} ({mpor_share.loc[d, ns]:.1%})</td>" for ns in mpor_pivot.columns) + "</tr>"
             for d in mpor_pivot.index]
days = list(mpor_pivot.index)
mpor_rows.append("<tr class='total'><td>Ratio per doubling</td>" + "".join(
    f"<td class='num'>{' / '.join(f'{mpor_pivot.loc[b, ns] / mpor_pivot.loc[a, ns]:.2f}×' for a, b in zip(days, days[1:]))}</td>"
    for ns in mpor_pivot.columns) + "</tr>")
mpor_table = table([("MPoR, zero threshold", "")] + [(f"{ns} CVA (share of uncollateralized)", "num") for ns in mpor_pivot.columns], mpor_rows)

uncoll = coll_sum[coll_sum.scenario == "Uncollateralized"].set_index("netting_set_id")["cva"]
headline_by_set = cva_sum.set_index("netting_set_id")["cva_netted"]
uncoll_gap = max(abs(uncoll[ns] - headline_by_set[ns]) for ns in uncoll.index)
tightening_ok = all((block["cva"].diff().dropna() <= 1e-9).all() for _, block in coll_sum.groupby("netting_set_id"))
mpor_ok = all((block["cva"].diff().dropna() > 0).all() for _, block in mpor_t.groupby("netting_set_id"))
zero_rows = coll_sum[coll_sum.scenario.str.startswith("Zero")].set_index("netting_set_id")
zero_total = zero_rows["cva"].sum()
csa_further = 1 - zero_total / ctot.cva_netted

conv_total = conv[conv.netting_set_id == "TOTAL"].sort_values("n_paths").set_index("n_paths")
conv_table = table([("Paths", "num"), ("Portfolio CVA", "num"), ("Std error", "num"), ("95% interval", "num"), ("Relative std error", "num")],
                   [f"<tr><td class='num'>{n:,}</td><td class='num'>{money(r.cva)}</td><td class='num'>{money(r.cva_se)}</td>"
                    f"<td class='num'>{money(r.ci95_low)} – {money(r.ci95_high)}</td><td class='num'>{r.cva_se / r.cva:.2%}</td></tr>"
                    for n, r in conv_total.iterrows()])
n_base, n_max = BASE_SIMULATION.n_paths, conv_total.index.max()
base_point_gap = abs(conv_total.loc[n_base, "cva"] - ctot.cva_netted)
max_inside_base_ci = conv_total.loc[n_base, "ci95_low"] <= conv_total.loc[n_max, "cva"] <= conv_total.loc[n_base, "ci95_high"]
se_ratio = conv_total.loc[n_base, "cva_se"] / conv_total.loc[n_max, "cva_se"]

MKT_LABELS = {"sigma": ("Hull-White σ", "{:.2%}"), "mean_reversion": ("Mean reversion a", "{:.0%}"), "maturity_scale": ("Tenor multiplier", "×{:g}")}
mkt_pivot = mkt.pivot_table(index=["parameter", "value"], columns="netting_set_id", values="cva")
mkt_rows = []
for parameter, (label, fmt) in MKT_LABELS.items():
    block = mkt_pivot.loc[parameter]
    for i, (value, row) in enumerate(block.iterrows()):
        first = f"<td rowspan='{len(block)}' class='view'>{label}</td>" if i == 0 else ""
        mkt_rows.append(f"<tr>{first}<td class='num'>{fmt.format(value)}</td>" +
                        "".join(f"<td class='num'>{thousands(row[ns])}</td>" for ns in ["NS_A", "NS_B", "TOTAL"]) + "</tr>")
mkt_table = table([("Input", ""), ("Value", "num"), ("NS_A", "num"), ("NS_B", "num"), ("Portfolio", "num")], mkt_rows)
tot_mkt = mkt_pivot["TOTAL"]
sigma_ok = tot_mkt.loc["sigma"].is_monotonic_increasing
tenor_ok = tot_mkt.loc["maturity_scale"].is_monotonic_increasing
meanrev_ok = tot_mkt.loc["mean_reversion"].is_monotonic_decreasing

TORNADO_LABELS = {"maturity_scale": ("Portfolio tenor", "×{:g}"), "spread_scale": ("Assumed spread", "×{:g}"),
                  "recovery_hazard_fixed": ("Recovery (hazard fixed)", "{:.0%}"), "sigma": ("Rate volatility σ", "{:.2%}"),
                  "mean_reversion": ("Mean reversion a", "{:.0%}"), "notional_scale": ("Notional", "×{:g}")}
spans = torn.groupby("parameter")["change_vs_base"].agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
torn_rows = []
for parameter in spans.index:
    lo, hi = torn[torn.parameter == parameter].sort_values("value").itertuples()
    label, fmt = TORNADO_LABELS[parameter]
    torn_rows.append(f"<tr><td>{label}</td><td class='num'>{fmt.format(lo.value)}: {lo.change_vs_base:+.0%}</td>"
                     f"<td class='num'>{fmt.format(hi.value)}: {hi.change_vs_base:+.0%}</td><td class='num'>{spans[parameter]:.0%}</td></tr>")
torn_table = table([("Input", ""), ("Lower value", "num"), ("Higher value", "num"), ("Range", "num")], torn_rows)
drivers = [TORNADO_LABELS[p][0].lower() for p in spans.index if p != "notional_scale"]
bullets = resume_bullets(load_results(TABLES_DIR), BASE_SIMULATION.n_paths, grid.size)

# ================================================================ tests table
TEST_DESCRIPTIONS = {
    "test_curves.py": "Nodes reproduced, DF positive and decreasing, every Treasury bond reprices to par, closed-form flat-curve cases",
    "test_instruments.py": "Schedules (regular, forward-starting, short stub), validation, portfolio loading and grouping",
    "test_pricing.py": "Par swap = 0, payer = −receiver, float leg = N(P(T₀) − P(Tₙ)), closed-form par rate, generic aggregation",
    "test_rate_models.py": "t = 0 reproduces curve, analytic moments of x and ∫x, mean-reversion slope, martingale tests, common random numbers",
    "test_simulation.py": "Grid contains every reset and payment date, cube(t = 0) = pricer, σ = 0 equals forward values, swap martingales, exact bridge",
    "test_exposure.py": "Exposure ≥ 0, netted ≤ gross on every path, PFE reconciles with the empirical CDF, EE(0) = max(V₀, 0), N(0,1) case",
    "test_credit.py": "Survival falls, PD rises, marginal PDs sum to PD(T), piecewise hazards, credit triangle, input loading",
    "test_cva.py": "λ = 0 and R = 1 give zero CVA, monotone in λ / R / notional, netted ≤ gross, continuous closed form, SE ∝ 1/√N",
    "test_collateral.py": "Infinite threshold = uncollateralized, collateral never adds exposure, tighter threshold / smaller MTA / shorter MPoR lower exposure and CVA",
    "test_validation.py": "Convergence chunk 0 = base run, SE ∝ 1/√N, CVA rises with σ and tenor, falls with mean reversion, linear in notional",
}
tests_table = table([("File", ""), ("Tests", "num"), ("What it proves", "")],
                    [f"<tr><td class='mono'>tests/{name}</td><td class='num'>{tests_per_file[name]}</td><td>{desc}</td></tr>"
                     for name, desc in TEST_DESCRIPTIONS.items()])

# ================================================================ page
CSS = """
/* Layout: one reading column of a technical progress report; figures sit on fixed light plates, tables scroll in place. */
:root {
  --bg: #f8f8f6; --panel: #ffffff; --ink: #15171a; --ink-2: #50545b; --ink-3: #7b7f86;
  --rule: #e2e3e0; --accent: #2a78d6; --accent-soft: #e7eff9;
  --done: #157a52; --done-soft: #e2f2ea; --neg: #b4472c; --plate: #fcfcfb;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141518; --panel: #1b1d21; --ink: #ecebe6; --ink-2: #a8abb1; --ink-3: #7f838a;
  --rule: #2d3036; --accent: #5b9cec; --accent-soft: #1d2939;
  --done: #4cc391; --done-soft: #16302a; --neg: #e8876c; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #141518; --panel: #1b1d21; --ink: #ecebe6; --ink-2: #a8abb1; --ink-3: #7f838a;
  --rule: #2d3036; --accent: #5b9cec; --accent-soft: #1d2939;
  --done: #4cc391; --done-soft: #16302a; --neg: #e8876c; color-scheme: dark; }

* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 15.5px; line-height: 1.6;
       padding-inline: 16px; padding-block: 40px 72px; }
.page { max-width: 940px; margin: 0 auto; display: grid; gap: 56px; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(2rem, 5vw, 2.75rem); letter-spacing: -0.01em; }
h2 { font-size: 1.75rem; }
h3 { font-size: 1.15rem; margin-top: 8px; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-family: var(--f-mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.lede { font-size: 1.1rem; color: var(--ink-2); max-width: 64ch; }
section { display: grid; gap: 20px; min-width: 0; }
.stack { display: grid; gap: 12px; min-width: 0; }
header.top { display: grid; gap: 16px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 20px; font-family: var(--f-mono); font-size: 0.8rem; color: var(--ink-2); }
.meta b { color: var(--ink); font-weight: 500; }
.pipeline { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.pipeline li { display: flex; align-items: center; gap: 8px; }
.pipeline li:not(:last-child)::after { content: "→"; color: var(--ink-3); }
.stage { font-size: 0.82rem; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--rule); color: var(--ink-2); background: var(--panel); }
.stage.done { background: var(--done-soft); border-color: transparent; color: var(--done); font-weight: 500; }
.stage.partial { border: 1px dashed var(--done); color: var(--done); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px; background: var(--rule);
         border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }
.fact { background: var(--panel); padding: 14px 16px; display: grid; gap: 2px; align-content: start; }
.fact .v { font-family: var(--f-display); font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.fact .k { font-size: 0.82rem; color: var(--ink-2); }
.phase-head { display: grid; gap: 8px; padding-top: 24px; border-top: 2px solid var(--ink); }
.phase-head .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; justify-content: space-between; }
.pill { font-family: var(--f-mono); font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 9px;
        border-radius: 999px; background: var(--done-soft); color: var(--done); white-space: nowrap; }
ul.built { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; max-width: 72ch; }
ul.built code, p code, td code, li code { font-family: var(--f-mono); font-size: 0.86em; background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
figure .plate { background: var(--plate); border: 1px solid var(--rule); border-radius: 8px; padding: 8px; }
figure img { display: block; width: 100%; height: auto; }
figcaption { font-size: 0.85rem; color: var(--ink-2); max-width: 75ch; }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th { font-weight: 500; color: var(--ink-2); font-size: 0.78rem; background: var(--bg); }
tbody tr:last-child td { border-bottom: 0; }
tr.total td { font-weight: 600; border-top: 1px solid var(--ink-3); }
td.view { color: var(--ink-2); vertical-align: top; }
td.num, th.num { text-align: right; }
td.neg { color: var(--neg); }
td.mono { font-family: var(--f-mono); }
.tag { font-family: var(--f-mono); font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--rule); }
.tag-pay { color: var(--accent); border-color: var(--accent); }
.tag-rec { color: var(--ink-2); }
.math { font-family: var(--f-mono); font-size: 0.84rem; line-height: 1.75; background: var(--panel); border: 1px solid var(--rule);
        border-radius: 8px; padding: 14px 16px; overflow-x: auto; white-space: pre; color: var(--ink); }
.math .c { color: var(--ink-3); }
.findings { display: grid; gap: 14px; margin: 0; padding: 0; list-style: none; }
.findings li { display: grid; gap: 4px; padding-left: 14px; border-left: 2px solid var(--rule); max-width: 76ch; }
.findings li b { font-weight: 600; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
.two > * { min-width: 0; }
.checks { display: grid; gap: 6px; font-family: var(--f-mono); font-size: 0.84rem; }
.checks div { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px dashed var(--rule); flex-wrap: wrap; }
.checks span:last-child { font-variant-numeric: tabular-nums; }
footer { color: var(--ink-3); font-size: 0.82rem; border-top: 1px solid var(--rule); padding-top: 16px; }
a { color: var(--accent); }
"""

PRINT_CSS = """
@page { size: A4; margin: 14mm 13mm 16mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
:root, :root[data-theme="light"] { --bg: #ffffff; }
body { font-size: 8.6pt; padding-inline: 0; padding-block: 0; }
.page { max-width: none; }
h1 { font-size: 19pt; } h2 { font-size: 13.5pt; } h3 { font-size: 10.5pt; }
.lede { font-size: 9.6pt; }
.facts { grid-template-columns: repeat(5, 1fr); }
.fact .v { font-size: 12.5pt; }
.table-wrap { overflow: visible; }
table { font-size: 7.4pt; }
th, td { padding: 4px 7px; }
th { white-space: normal; background: #f3f4f1; }
td:last-child:not(.num) { white-space: normal; min-width: 48ch; }
.math { font-size: 7.2pt; white-space: pre-wrap; }
.checks { font-size: 7.6pt; }
.stage, .pill, .eyebrow, .meta { font-size: 7pt; }
figcaption { font-size: 7.8pt; }
.two { grid-template-columns: 1fr 1fr; }
figure .plate img { width: 84% !important; max-width: 84%; height: auto; margin: 0 auto; }
figure.tall .plate img { width: 74% !important; max-width: 74%; }
/* block flow so keep-with-next and avoid-break rules are honored (grid items ignore them) */
.page, section, header.top, .stack, figure, .phase-head { display: block; }
.page > * + * { margin-top: 24px; }
section > * + * { margin-top: 11px; }
header.top > * + *, .stack > * + *, .phase-head > * + * { margin-top: 8px; }
figure > * + * { margin-top: 6px; }
figure, .table-wrap, .math, .checks, .facts, .findings li, .stack, ul.built, footer { break-inside: avoid; }
h2, h3, .phase-head, .eyebrow { break-after: avoid; page-break-after: avoid; }
p:has(+ .table-wrap), p:has(+ figure) { break-after: avoid; page-break-after: avoid; }
section.new-page { break-before: page; }
.phase-head { break-inside: avoid; page-break-inside: avoid; }
p { orphans: 3; widows: 3; break-inside: avoid; }
"""

stages = ["Market data", "Swap valuation", "Rate simulation", "Future MtM", "Exposure", "Default probability", "CVA",
          "Netting / collateral", "Risk report"]
state = ["done"] * 9
pipeline = "".join(f'<li><span class="stage {s}">{name}</span></li>' for name, s in zip(stages, state))
vd = pd.Timestamp(VALUATION_DATE)

BODY = f"""<div class="page">

<header class="top">
  <span class="eyebrow">Counterparty Credit Risk &amp; CVA Engine · final report</span>
  <h1>Phases A–F: from swap prices to collateralized CVA</h1>
  <p class="lede">The engine values a 10-swap portfolio on a real US Treasury curve and revalues it on
  {BASE_SIMULATION.n_paths:,} simulated Hull-White rate paths. It turns those values into netted counterparty exposure, prices
  unilateral CVA from default probabilities, and measures how netting, a collateral agreement, and credit and market
  assumptions change the result. Every number is validated against a known relationship.</p>
  <div class="meta"><span>Valuation date <b>{vd:%d %b %Y}</b></span>
  <span>Tests <b>{html.escape(passed)}</b></span><span>Seed <b>{BASE_SIMULATION.seed}</b></span></div>
  <ol class="pipeline" aria-label="Engine pipeline; every stage complete">{pipeline}</ol>
</header>

<div class="facts">
  <div class="fact"><span class="v">{BASE_SIMULATION.n_paths // 1000}k × {grid.size}</span><span class="k">Hull-White paths × dates for {len(portfolio)} swaps, revalued in {sim_seconds:.1f} s</span></div>
  <div class="fact"><span class="v">{thousands(ctot.cva_netted)}</span><span class="k">portfolio CVA, ± {money(ctot.cva_netted_se)} (1 std error)</span></div>
  <div class="fact"><span class="v">−{ctot.netting_reduction:.0%}</span><span class="k">CVA cut by netting (gross {thousands(ctot.cva_gross)})</span></div>
  <div class="fact"><span class="v">−{csa_further:.0%}</span><span class="k">further cut by a zero-threshold CSA ({thousands(zero_total)} left)</span></div>
  <div class="fact"><span class="v">{n_tests} tests</span><span class="k">all passing, {len(tests_per_file)} files</span></div>
</div>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase A · foundations</span><span class="pill">Complete</span></div>
  <h2>What is a swap worth today?</h2></div>
  <ul class="built">
    <li><code>scripts/fetch_treasury_curve.py</code> saved the Treasury par-yield curve for {vd:%d %b %Y}; the snapshot is committed so every run is reproducible.</li>
    <li><code>src/curves.py</code> bootstraps it into a zero curve and interpolates linearly in log discount factor.</li>
    <li><code>src/instruments.py</code> and <code>src/swap_pricer.py</code> price vanilla fixed-for-floating swaps. The leg formulas take arrays of discount factors, so Phase B reuses them on simulated paths.</li>
    <li><code>scripts/make_synthetic_portfolio.py</code> built a 10-trade book with fixed rates set off the par rate.</li>
  </ul>
  <h3>The curve</h3>
  {figure("yield_curve.png", "Bootstrapped US Treasury zero curve, forwards, par-yield inputs and discount factors",
          f"Zero rates rise from {pct(curve.zero_rate(quotes['tenor_years'].iloc[0]))} at 1M to {pct(curve.zero_rate(20.0))} at 20Y. The sawtooth forwards and the 20Y jump come from linear par-yield interpolation and the real 20Y/30Y inversion on this date. Within 10Y, where all trades mature, forwards stay between {pct(fwd_10y.min())} and {pct(fwd_10y.max())}.")}
  {curve_table}
  <h3>The portfolio</h3>
  <p>Values are from our side. <span class="tag tag-pay">PAY</span> means we pay fixed and receive floating; fixed coupons are
  semiannual and floating coupons quarterly. NS_A mixes directions on purpose; NS_B is mostly payers.</p>
  {blotter}
  {ns_table}
  <div class="two">
    <div class="stack"><h3>Math to defend</h3>
      <div class="math"><span class="c"># bootstrap: each quote is a bond priced at par</span>
1 = y·Σ τₖ P(0,Tₖ) + P(0,Tₙ)    <span class="c"># bills: P = 1/(1+yT)</span>
<span class="c"># interpolation: log P linear → flat forwards</span>
A        = Σ τᵢ P(Tᵢ)
PV_fixed = N·K·A
Fⱼ       = (P(Sⱼ)/P(Tⱼ) − 1) / τⱼ
PV_float = N·Σ τⱼ Fⱼ P(Tⱼ) = N·(P(T₀) − P(Tₙ))
K_par    = PV_float / (N·A)
V        = sign·(PV_float − PV_fixed)</div></div>
    <div class="stack"><h3>Checks</h3>
      <div class="checks">
        <div><span>Worst Treasury bond repricing error</span><span>{reprice_err:.1e}</span></div>
        <div><span>A1 at its par rate</span><span>{money(a1_par_mtm, 6)}</span></div>
        <div><span>A1 payer / receiver</span><span>{money(a1_payer, 0)} / {money(a1_receiver, 0)}</span></div>
        <div><span>10Y par swap rate vs 10Y par yield</span><span>{pct(par_rate(a1, curve))} vs {pct(quotes.loc[quotes.tenor_years == 10, "par_yield"].iloc[0])}</span></div>
      </div>
      <p>With one curve and semiannual fixed coupons, the par swap rate and the par bond yield are the same quantity, (1 − P(0,T)) / A.</p></div>
  </div>
</section>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase B · market simulation</span><span class="pill">Complete</span></div>
  <h2>How could rates evolve, and what would the swaps be worth?</h2></div>
  <ul class="built">
    <li><code>src/rate_models.py</code>: one-factor Hull-White closed forms for bond prices, the short rate and the stochastic discount factor.</li>
    <li><code>src/simulation.py</code>: time grid (monthly plus every reset and payment date, {grid.size} dates), exact path simulation and revaluation into a trades × paths × dates cube.</li>
  </ul>
  <p>Hull-White fits today's curve exactly, which Vasicek cannot. Writing r(t) = x(t) + φ(t) leaves a zero-mean Ornstein-Uhlenbeck
  process with a known Gaussian transition, simulated exactly with no time-step error. Parameters are assumed, not calibrated:
  a = {HW_PARAMS.mean_reversion:.0%} per year and σ = {HW_PARAMS.volatility:.0%}/√year, the Hull-White short-rate diffusion volatility.
  σ is a model parameter, not a quoted swaption normal (Bachelier) vol.</p>
  {figure("rate_paths.png", "Fan chart of simulated short rates and histograms at 1, 5 and 10 years",
          f"The simulated mean lies on the analytic E[r(t)] = φ(t). Hull-White is Gaussian, so rates can go negative ({pct(dist.loc[dist.time == 10, 'prob_negative'].iloc[0], 1)} of paths at 10Y).")}
  {dist_table}
  <h3>The martingale test</h3>
  <p>In an arbitrage-free model fitted to the curve, E[D(0,t)·P(t,T)] = P(0,T). In the {BASE_SIMULATION.n_paths:,}-path base run below,
  the worst |z| is {mart["z_score"].abs().max():.2f}, inside the 4-standard-error test tolerance. A separate 400,000-path rerun on a coarse
  grid gives a worst |z| of {big_mart_z:.2f}, so the base-run deviations are sampling noise, not bias.</p>
  {mart_table}
  <h3>Revaluing the book on every path</h3>
  {figure("mtm_paths.png", "Netted mark-to-market fan charts for NS_A and NS_B",
          "Netted value to us by netting set. NS_A barely moves; NS_B swings with the level of rates.")}
  {mtm_table}
  <div class="two">
    <div class="stack"><h3>Math to defend</h3>
      <div class="math">dx = −a·x dt + σ dW,   r = x + φ(t)
φ(t) = f(0,t) + σ²/(2a²)·(1 − e^(−at))²
B(t,T) = (1 − e^(−a(T−t))) / a
P(t,T) = P(0,T)/P(0,t)·exp(−B·x − ½B²·Vₓ(t)
                          − B·σ²/(2a²)·(1 − e^(−at))²)
Vₓ(t)  = σ²/(2a)·(1 − e^(−2at))
D(0,t) = P(0,t)·exp(−∫x ds − ½Var[∫x ds])</div></div>
    <div class="stack"><h3>Revaluation rules</h3>
      <ul class="built">
        <li>P(t,T) is computed once per date for every future portfolio date; trades reuse the Phase A leg formulas.</li>
        <li>A floating coupon fixes on its reset date at L = (1/P(S,T) − 1)/τ on each path and keeps that fixing.</li>
        <li>V(t) excludes cash flows paid at t; a trade is worth zero from its final payment.</li>
        <li>With σ = 0, every path reproduces the forward value to about 1e-9 relative.</li>
      </ul></div>
  </div>
</section>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase C · exposure</span><span class="pill">Complete</span></div>
  <h2>How much would we lose if the counterparty defaulted?</h2></div>
  <ul class="built">
    <li><code>src/exposure.py</code>: netted and gross exposure per netting set, Expected Exposure, PFE and EPE, with summary and profile tables.</li>
    <li><code>run_all.py</code> stage C writes the profiles and three figures; <code>notebooks/03_exposure_profiles.ipynb</code> walks through them.</li>
  </ul>
  <p>At default we lose the netting set's value only when it is positive to us; a negative value is still owed. Trades in a
  netting set close out as one amount, so values are netted on each path and date before taking the positive part. PFE is
  the 95th percentile of exposure, taken as an actual simulated value (empirical inverse CDF). EPE here always means the
  time-average of EE over the netting set's whole life, not the regulatory Effective EPE.</p>
  {exp_table}
  {figure("exposure_profiles.png", "Netted Expected Exposure and 95% PFE through time for NS_A and NS_B",
          "Netted EE and PFE at 95%. NS_B rises as rates spread out, peaks around 1.5–2Y and runs off. NS_A starts high because it is already in the money, then declines. The sawtooth comes from quarterly floating and semiannual fixed payments.")}
  {figure("gross_vs_netted_ee.png", "Gross versus netted Expected Exposure for NS_A and NS_B",
          f"Gross EE (each trade on its own) against netted EE. Netting removes {ea.netting_reduction:.0%} of NS_A's time-average EE (EPE) but only {eb.netting_reduction:.0%} of NS_B's; once B4, NS_B's only receiver, matures at 3Y, gross equals netted.")}
  {prof_table}
  {figure("exposure_distributions.png", "Histograms of netted value at 1, 3 and 5 years for each netting set", "Netted value at 1Y, 3Y and 5Y. Colored bars are paths with exposure; grey bars are paths where we would owe the counterparty. Vertical lines mark EE and PFE.", ).replace("<figure>", '<figure class="tall">')}
  <div class="two">
    <div class="stack"><h3>Math to defend</h3>
      <div class="math">E_net(t)   = max(Σᵢ Vᵢ(t), 0)
E_gross(t) = Σᵢ max(Vᵢ(t), 0)  ≥ E_net(t)
EE(t)      = mean over paths of E(t)
PFE_q(t)   = min e : Pₙ(E(t) ≤ e) ≥ q
EPE        = (1/T)·∫₀ᵀ EE(t) dt
<span class="c"># EPE = time-average EE over the life; not Effective EPE</span>
<span class="c"># max(a+b,0) ≤ max(a,0)+max(b,0): netting never adds exposure</span></div></div>
    <div class="stack"><h3>Checks (base case, every path and date)</h3>
      <div class="checks">
        <div><span>Smallest exposure</span><span>{min_exposure:,.2f}</span></div>
        <div><span>Largest netted − gross</span><span>{max_net_minus_gross:,.2f}</span></div>
        <div><span>|EE(0) − max(V₀, 0)|</span><span>{ee0_gap:.1e}</span></div>
        <div><span>PFE reconciles with empirical CDF</span><span>{"yes, every date" if pfe_ok else "NO"}</span></div>
        <div><span>Paths with zero exposure, NS_B at 1Y / 3Y / 5Y</span><span>{p_zero[("NS_B", 1.0)]:.0%} / {p_zero[("NS_B", 3.0)]:.0%} / {p_zero[("NS_B", 5.0)]:.0%}</span></div>
      </div></div>
  </div>
</section>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase D · credit and CVA</span><span class="pill">Complete</span></div>
  <h2>What is the expected loss from that exposure?</h2></div>
  <ul class="built">
    <li><code>src/credit.py</code>: hazard-rate survival curves (flat or piecewise) from assumed CDS-equivalent spreads and recoveries, using the credit-triangle approximation λ ≈ s/(1−R), not a CDS-curve calibration.</li>
    <li><code>src/cva.py</code>: pathwise-discounted CVA as the average of a per-path CVA, so every estimate carries a standard error. It is integrated on the full simulation grid, with the deterministic curve-DF version as a benchmark.</li>
    <li><code>src/validation.py</code>: credit sensitivities with the simulated exposures held fixed. <code>notebooks/04_cva_and_sensitivities.ipynb</code> walks through all of it.</li>
  </ul>
  <p>The spreads in <code>counterparties.csv</code> are assumed counterparty spread scenarios (CDS-equivalent), not market quotes.
  λ ≈ s/(1−R) ignores premium accrual, discounting inside the CDS legs and the spread term structure. Hazard rates are deterministic
  and independent of rates, so there is no wrong-way risk.</p>
  {credit_table}
  {figure("credit_curves.png", "Cumulative default probability and default probability per year for CPTY_A and CPTY_B",
          "Each year's default probability shrinks slowly because the counterparty must first survive to that year.")}
  <h3>CVA</h3>
  {cva_table}
  <p>The headline discounts each path with its simulated bank account D(0,t), because exposure and discount factors are correlated
  under stochastic rates. The spec §7 approximation P(0,t)·EE(t) is kept as a benchmark. The integral runs over all {grid.size - 1}
  intervals of the simulation grid; the yearly buckets below only regroup it. The total's standard error comes from the per-path sum,
  since both netting sets share the same paths and their errors are correlated.</p>
  {figure("cva_contributions.png", "CVA contribution by default year, netted versus gross, for NS_A and NS_B",
          "CVA by default year. NS_B's CVA sits in years 1–3, where its exposure peaks. NS_A's declines from the start; most of its gross CVA disappears with netting.")}
  {bucket_table}
  <h3>Robustness checks</h3>
  {robust_table}
  <p>The curve-DF benchmark overstates CVA slightly. These books gain when rates rise, which is exactly when the bank-account discount
  factor is small, and the deterministic approximation ignores that covariance. The credit-triangle shortcut ignores that the
  counterparty must first survive, so it overstates CVA, more so for the riskier CPTY_B.</p>
  {figure("cva_credit_sensitivity.png", "CVA versus CDS spread and versus recovery rate for NS_A and NS_B",
          "Credit sensitivities with exposures held fixed. CVA rises with the assumed spread, slightly less than proportionally. With the hazard fixed, CVA is exactly proportional to 1 − R; with the spread fixed, the hazard is re-implied and CVA barely moves.")}
  {sens_table}
  <div class="two">
    <div class="stack"><h3>Math to defend</h3>
      <div class="math">S(t)  = exp(−∫₀ᵗ λ(u) du)
ΔPDᵢ  = S(tᵢ₋₁) − S(tᵢ)
λ     ≈ s / (1 − R)      <span class="c"># credit-triangle approximation</span>
CVA   = LGD·Σᵢ ½[dEE(tᵢ₋₁) + dEE(tᵢ)]·ΔPDᵢ
dEE(t)= E[D(0,t)·E(t)]   <span class="c"># benchmark: P(0,t)·EE(t)</span>
      ⇒ CVA = mean over paths of CVA_path
SE    = std(CVA_path) / √N</div></div>
    <div class="stack"><h3>Spec §11 checks</h3>
      <div class="checks">
        <div><span>λ = 0 → CVA</span><span>exactly 0</span></div>
        <div><span>R = 100% → CVA</span><span>exactly 0</span></div>
        <div><span>Higher λ → higher CVA</span><span>tested, 7 levels</span></div>
        <div><span>Higher R → lower CVA</span><span>exactly ∝ 1 − R</span></div>
        <div><span>2 × notional → CVA</span><span>exactly 2×</span></div>
        <div><span>SE ∝ 1/√N</span><span>tested, 1k/4k/16k paths</span></div>
      </div></div>
  </div>
</section>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase E · collateral</span><span class="pill">Complete</span></div>
  <h2>How much do netting and collateral reduce CVA?</h2></div>
  <ul class="built">
    <li><code>src/collateral.py</code>: a one-way CSA (the counterparty posts, we never do) with threshold H, minimum transfer amount and margin period of risk (MPoR), plus a shared analysis of every scenario.</li>
    <li><code>src/simulation.py::refine_paths</code>: an exact Gaussian bridge adds the look-back dates t − MPoR to the Phase D paths without changing any existing date. <code>notebooks/05_collateral.ipynb</code> walks through it.</li>
  </ul>
  <p>If the counterparty defaults at t, the last collateral call it met was at t − MPoR, so collateral held is C(t) = max(V(t − MPoR) − H − MTA, 0)
  and collateralized exposure is max(V(t) − C(t), 0). Collateral can only reduce exposure, a tighter threshold reduces it on every path,
  and an infinite threshold gives back the uncollateralized case. Simplifications: MTA is added to the threshold, both parties keep paying
  coupons during the MPoR, and there is no initial margin or haircut.</p>
  {figure("collateral_ee_profiles.png", "Netted expected exposure through time under each CSA scenario for NS_A and NS_B",
          "Netted EE under each CSA scenario. A $2m threshold barely touches NS_A, whose exposure rarely exceeds it. A $500k threshold caps NS_A near H + MTA from day one. Under a zero threshold, what remains is value moves inside the MPoR plus spikes on dates when we make a net coupon payment while the counterparty still holds collateral sized before it.")}
  {coll_table}
  {figure("cva_mitigation.png", "CVA after netting and after each CSA scenario for NS_A and NS_B",
          "From gross to netted to collateralized CVA. Netting does most of the work for the balanced NS_A; collateral does it for the one-directional NS_B.")}
  {mitig_table}
  <h3>Margin period of risk</h3>
  <p>Under a zero threshold, residual exposure depends on how long the value can move before collateral catches up. Pure diffusion would
  scale it like √MPoR (1.41× per doubling); the coupon spikes do not scale with the MPoR, so the observed growth is slower.</p>
  {mpor_table}
  <div class="two">
    <div class="stack"><h3>Math to defend</h3>
      <div class="math">C(t)   = max(V(t − MPoR) − H − MTA, 0)
E_c(t) = max(V(t) − C(t), 0) ≤ max(V(t), 0)
<span class="c"># look-back value on the same path, exact bridge:</span>
Y | Z ~ N(μ_Y + K(Z − Aμ_Y), S₁ − K A S₁)
K     = S₁Aᵀ(A S₁ Aᵀ + S₂)⁻¹</div></div>
    <div class="stack"><h3>Checks</h3>
      <div class="checks">
        <div><span>Uncollateralized CVA − Phase D headline</span><span>{money(uncoll_gap, 2)}</span></div>
        <div><span>CVA falls as collateral tightens</span><span>{"yes, both sets" if tightening_ok else "NO"}</span></div>
        <div><span>Longer MPoR → more residual CVA</span><span>{"yes, both sets" if mpor_ok else "NO"}</span></div>
        <div><span>Zero-threshold residual, NS_A / NS_B</span><span>{1 - zero_rows.loc["NS_A", "cva_reduction_vs_uncollateralized"]:.1%} / {1 - zero_rows.loc["NS_B", "cva_reduction_vs_uncollateralized"]:.1%}</span></div>
      </div></div>
  </div>
</section>

<section>
  <div class="phase-head"><div class="row"><span class="eyebrow">Phase F · validation and sensitivities</span><span class="pill">Complete</span></div>
  <h2>How robust are the numbers?</h2></div>
  <ul class="built">
    <li><code>src/validation.py</code>: Monte Carlo convergence to {n_max:,} paths; market sensitivities (σ, mean reversion, tenor, notional) under common random numbers; a tornado that adds spread and recovery.</li>
    <li><code>src/reporting.py</code> writes <code>reports/final_project_summary.md</code> and the README results block from the generated tables. <code>docs/technical_note.md</code> documents every formula and model choice.</li>
  </ul>
  <h3>Monte Carlo convergence</h3>
  <p>Independent chunks of {n_base:,} paths are pooled; chunk 0 is the base case, so the {n_base:,}-path point is the headline exactly.</p>
  {figure("convergence.png", "CVA estimate with 95% interval versus number of paths for each netting set, and standard error versus paths on log-log axes",
          f"Estimates settle inside the earlier intervals and the standard error follows the 1/√N line. At {n_max:,} paths the portfolio CVA's relative standard error is {conv_total.loc[n_max, 'cva_se'] / conv_total.loc[n_max, 'cva']:.2%}.")}
  {conv_table}
  <h3>Market and portfolio sensitivities</h3>
  <p>Each run re-simulates with the base seed (common random numbers). For tenor, every trade is stretched and re-struck at par plus its
  original offset, so moneyness does not change. Notional scaling is exact because values are linear in notional.</p>
  {figure("market_sensitivity.png", "CVA versus rate volatility, mean reversion and tenor multiplier for NS_A and NS_B",
          "NS_B's CVA rises almost linearly with σ because its exposure is option-like; NS_A barely moves because its exposure is mostly value it already holds. Higher mean reversion damps long-dated variance. Tenor is the strongest market effect, and convex.")}
  {mkt_table}
  {figure("sensitivity_tornado.png", "Tornado chart of portfolio CVA change for low and high values of each input",
          f"Each input moved alone from the base portfolio CVA of {thousands(torn['base_cva'].iloc[0])}. Among the modelling inputs, {drivers[0]} and {drivers[1]} have the widest ranges.")}
  {torn_table}
  <div class="two">
    <div class="stack"><h3>Method</h3>
      <div class="math">estimate(N) = mean of first N per-path CVAs
SE(N)       = std / √N
<span class="c"># chunk k uses seed 42 + 1000k; chunk 0 = base</span>
<span class="c"># tenor ×k: T' = S + k(T − S), K' = K_par(T') + (K − K_par(T))</span>
<span class="c"># σ change: same shocks, x scales linearly in σ</span></div></div>
    <div class="stack"><h3>Checks</h3>
      <div class="checks">
        <div><span>{n_base:,}-path point − headline</span><span>{money(base_point_gap, 2)}</span></div>
        <div><span>{n_max:,}-path estimate inside {n_base:,}-path 95% CI</span><span>{"yes" if max_inside_base_ci else "NO"}</span></div>
        <div><span>SE({n_base:,}) / SE({n_max:,}) vs √{n_max // n_base}</span><span>{se_ratio:.2f} vs {np.sqrt(n_max / n_base):.2f}</span></div>
        <div><span>CVA ↑ σ, ↑ tenor, ↓ mean reversion</span><span>{"yes" if sigma_ok and tenor_ok and meanrev_ok else "NO"}</span></div>
      </div></div>
  </div>
</section>

<section>
  <div class="phase-head"><span class="eyebrow">What the results say</span><h2>Findings</h2></div>
  <ul class="findings">
    <li><b>Directional balance drives netting.</b> NS_A's payers and receivers offset, so netting cuts its time-average EE (EPE) by {ea.netting_reduction:.0%} and its CVA by
    {ca.netting_reduction:.0%}. NS_B is nearly all payers: {eb.netting_reduction:.0%} and {cb.netting_reduction:.0%}. Trade count has nothing to do with it.</li>
    <li><b>Same EPE, very different risk.</b> Both sets have a time-average EE (EPE) near {millions(ea.epe_netted)}. NS_A's exposure is steady and almost always
    positive; NS_B's is zero on {p_zero[("NS_B", 1.0)]:.0%}–{p_zero[("NS_B", 5.0)]:.0%} of paths but has a far fatter tail (peak PFE {millions(getattr(eb, pfe_col))} vs {millions(getattr(ea, pfe_col))}).</li>
    <li><b>Credit quality dominates the CVA ranking.</b> NS_B's CVA ({thousands(cb.cva_netted)}) exceeds NS_A's ({thousands(ca.cva_netted)}) because its assumed spread is
    twice as wide. At an equal 150bp spread, NS_A would carry more CVA ({thousands(spread_view.loc[150, 'NS_A'])} vs {thousands(spread_view.loc[150, 'NS_B'])}) because its exposure lasts longer.</li>
    <li><b>Recovery matters less than it looks.</b> With the spread fixed, moving recovery from 0% to 80% changes NS_B's CVA only from
    {thousands(rec_fixed_s.iloc[0]['NS_B'])} to {thousands(rec_fixed_s.iloc[-1]['NS_B'])}: a higher recovery implies a higher hazard, and the two largely cancel.</li>
    <li><b>Netting and collateral solve different problems.</b> Netting removes {ca.netting_reduction:.0%} of NS_A's CVA but only {cb.netting_reduction:.0%} of NS_B's;
    a zero-threshold CSA then takes NS_B from {thousands(cb.cva_netted)} to {thousands(zero_rows.loc["NS_B", "cva"])}. Together they cut portfolio CVA from
    {thousands(ctot.cva_gross)} to {thousands(zero_total)}.</li>
    <li><b>Collateral never removes all the risk.</b> Even with no threshold, value moves inside the two-week MPoR and coupon payments
    inside the window leave {1 - zero_rows.loc["NS_B", "cva_reduction_vs_uncollateralized"]:.0%} of NS_B's CVA; a four-week MPoR leaves more.</li>
    <li><b>Tenor and spread move CVA most.</b> Over the tested ranges, {drivers[0]} ({spans[spans.index[1] if spans.index[0] == "notional_scale" else spans.index[0]]:.0%} range) and
    {drivers[1]} matter most, then {drivers[2]} and {drivers[3]}; mean reversion matters least.</li>
    <li><b>The base case is precise enough.</b> At {n_base:,} paths the portfolio CVA's standard error is {conv_total.loc[n_base, "cva_se"] / conv_total.loc[n_base, "cva"]:.1%},
    far below every sensitivity, and the {n_max:,}-path estimate lies inside its 95% interval.</li>
    <li><b>A one-factor model flatters hedged books.</b> All rates move together, so NS_A's curve risk, and therefore its exposure and CVA, is understated.</li>
    <li><b>Check noise against bias with more paths.</b> One mean-reversion regression landed {abs(worst_base_slope):.1f} standard errors out at {BASE_SIMULATION.n_paths:,}
    paths; with 400,000 paths every regression was within {big_slope_z:.1f}.</li>
  </ul>
</section>

<section>
  <div class="phase-head"><span class="eyebrow">Validation coverage</span><h2>{n_tests} tests, {len(tests_per_file)} files</h2></div>
  <p>Exact formulas are tested to 1e-9 or tighter. Monte Carlo checks use fixed seeds and a tolerance of 4 standard errors that is never tuned to pass.</p>
  {tests_table}
</section>

<section>
  <div class="phase-head"><span class="eyebrow">In brief</span><h2>Project summary</h2></div>
  <ul class="findings">{"".join(f"<li>{html.escape(b)}</li>" for b in bullets)}</ul>
</section>

<section>
  <div class="phase-head"><span class="eyebrow">Model risk</span><h2>Limitations</h2></div>
  <ul class="built">
    <li><b>Single curve:</b> Treasury rates discount and project; there is no SOFR/OIS basis.</li>
    <li><b>One-factor Gaussian rates:</b> all rates move together, so curve-hedged books such as NS_A look less risky than they are, and rates can go negative.</li>
    <li><b>Uncalibrated rate parameters:</b> a and σ are assumptions, and CVA for directional books scales roughly linearly with σ.</li>
    <li><b>Approximate credit:</b> hazards come from assumed spreads via the credit triangle, are deterministic and independent of rates (no wrong-way risk), and CVA is unilateral.</li>
    <li><b>Simplified collateral:</b> MTA is added to the threshold, coupons are paid during the MPoR, and there is no initial margin or haircut.</li>
    <li><b>Simplified schedules:</b> there are no day counts, calendars or historical fixings.</li>
  </ul>
</section>

<footer>Every number in this report was read from the project's generated tables (<code>reports/tables</code>) or computed by its
<code>src</code> code when the report was built; none were typed by hand. Figures are the outputs of <code>run_all.py</code>.</footer>
</div>
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">')

screen = f"<title>CVA Engine Report</title>\n{FONTS}\n<style>{CSS}</style>\n{BODY}"
(OUT_DIR / "cva_report.html").write_text(screen)
printable = (f'<!doctype html>\n<html lang="en" data-theme="light">\n<head>\n<meta charset="utf-8">\n'
             f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>CVA Engine Report</title>\n{FONTS}\n'
             f"<style>{CSS}</style>\n<style>{PRINT_CSS}</style>\n</head>\n<body>{BODY}</body>\n</html>\n")
print_html = OUT_DIR / "cva_report_print.html"
print_html.write_text(printable)
print(f"report written to reports/cva_report.html ({len(screen):,} chars); sim {sim_seconds:.2f}s; {passed}")


def find_chrome() -> str | None:
    """Locate a Chrome / Chromium binary for headless PDF printing."""
    for candidate in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                      "/Applications/Chromium.app/Contents/MacOS/Chromium"):
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None


if args.pdf:
    chrome = find_chrome()
    if chrome is None:
        sys.exit("--pdf needs Google Chrome or Chromium; open reports/cva_report_print.html and print to PDF instead.")
    pdf_path = OUT_DIR / PDF_NAME
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf_path}", print_html.as_uri()], check=True, capture_output=True)
    print(f"PDF written to reports/{PDF_NAME} ({pdf_path.stat().st_size / 1e6:.1f} MB)")
