"""Shared figure style and project charts.

All figures use one style: a light surface, recessive grid and axes, thin lines, and
a fixed categorical color order. Colors follow the entity, never its rank. Each panel
has a single y-axis; two measures with different scales go in separate panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.credit import CounterpartyCredit
from src.curves import DiscountCurve

# Categorical slots in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "font.family": "sans-serif",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
}


def apply_style() -> None:
    """Apply the project figure style to matplotlib."""
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure (creating the folder) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_curve(curve: DiscountCurve, par_yields: pd.DataFrame | None = None, max_time: float = 30.0) -> plt.Figure:
    """Two panels: rates (par quotes, zero rates, instantaneous forwards) and discount factors."""
    apply_style()
    grid = np.linspace(0.0, max_time, 1201)
    fig, (ax_rates, ax_df) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_rates.plot(grid, 100 * curve.zero_rate(grid), color=SERIES[0], label="Zero rate (cont. comp.)")
    ax_rates.step(grid, 100 * curve.inst_forward(grid), where="post", color=SERIES[1], linewidth=1.2,
                  label="Instantaneous forward")
    if par_yields is not None:
        ax_rates.plot(par_yields["tenor_years"], par_yields["par_yield_pct"], linestyle="none", marker="o",
                      markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, color=SERIES[2],
                      label="Treasury par yield (input)")
    ax_rates.set_title("Bootstrapped US Treasury curve")
    ax_rates.set_xlabel("Maturity (years)")
    ax_rates.set_ylabel("Rate (%)")
    ax_rates.set_xlim(0, max_time)
    ax_rates.legend(loc="lower right")

    ax_df.plot(grid, curve.df(grid), color=SERIES[0])
    ax_df.set_title("Discount factor P(0, t)")
    ax_df.set_xlabel("Maturity (years)")
    ax_df.set_ylabel("Discount factor")
    ax_df.set_xlim(0, max_time)
    ax_df.set_ylim(0, 1.02)

    fig.tight_layout()
    return fig


def _fan(ax: plt.Axes, times: np.ndarray, values: np.ndarray, color: str, n_sample: int, scale: float = 1.0,
         quantiles: tuple[float, float] = (5, 95)) -> None:
    """Sample paths, a percentile band and the cross-path mean on one axis."""
    lower, upper = np.percentile(values, quantiles, axis=0)
    for path in values[:n_sample]:
        ax.plot(times, scale * path, color=color, linewidth=0.5, alpha=0.25)
    ax.fill_between(times, scale * lower, scale * upper, color=color, alpha=0.15, linewidth=0,
                    label=f"{quantiles[0]:g}–{quantiles[1]:g}% band")
    ax.plot(times, scale * values.mean(axis=0), color=color, linewidth=2.0, label="Mean")


def plot_rate_paths(times: np.ndarray, short_rates: np.ndarray, analytic_mean: np.ndarray,
                    snapshot_times: tuple[float, ...] = (1.0, 5.0, 10.0), n_sample: int = 25) -> plt.Figure:
    """Short-rate fan chart (left) and the simulated r(t) distribution at a few horizons (right)."""
    apply_style()
    fig, (ax_fan, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    _fan(ax_fan, times, short_rates, SERIES[0], n_sample, scale=100)
    ax_fan.plot(times, 100 * analytic_mean, color=SERIES[1], linewidth=1.4, linestyle="--",
                label="Analytic E[r(t)] = φ(t)")
    ax_fan.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax_fan.set_title("Simulated Hull-White short rate")
    ax_fan.set_xlabel("Time (years)")
    ax_fan.set_ylabel("Short rate (%)")
    ax_fan.set_xlim(times[0], times[-1])
    ax_fan.legend(loc="upper left", ncol=3)

    bins = np.linspace(np.percentile(short_rates, 0.1), np.percentile(short_rates, 99.9), 60)
    for color, t in zip(SERIES, snapshot_times):
        k = int(np.argmin(np.abs(times - t)))
        ax_hist.hist(100 * short_rates[:, k], bins=100 * bins, density=True, histtype="step", linewidth=1.8,
                     color=color, label=f"t = {t:g}Y")
    ax_hist.set_title("Distribution of r(t)")
    ax_hist.set_xlabel("Short rate (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_mtm_fan(times: np.ndarray, mtm_by_set: dict[str, np.ndarray], n_sample: int = 25) -> plt.Figure:
    """One panel per netting set: netted MtM sample paths, 5–95% band and mean, in $m."""
    apply_style()
    fig, axes = plt.subplots(1, len(mtm_by_set), figsize=(5.5 * len(mtm_by_set), 4.2), sharey=True, squeeze=False)
    for ax, (netting_set, mtm) in zip(axes[0], mtm_by_set.items()):
        _fan(ax, times, mtm, SERIES[0], n_sample, scale=1e-6)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(f"Netted MtM — {netting_set}")
        ax.set_xlabel("Time (years)")
        ax.set_xlim(times[0], times[-1])
    axes[0][0].set_ylabel("Portfolio value to us ($m)")
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    return fig


# Exposure charts use one color per quantity throughout: netted EE = SERIES[0],
# PFE = SERIES[1], gross EE = SERIES[2]. Values are shown in $m.
NEUTRAL_FILL = "#c9c8c2"


def _panels(n: int, height: float = 4.0, sharey: bool = True) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, height), sharey=sharey, squeeze=False)
    return fig, axes[0]


def plot_exposure_profiles(times: np.ndarray, ee_by_set: dict[str, np.ndarray], pfe_by_set: dict[str, np.ndarray],
                           quantile: float = 0.95) -> plt.Figure:
    """One panel per netting set: netted Expected Exposure and PFE through time."""
    apply_style()
    fig, axes = _panels(len(ee_by_set))
    for ax, ns in zip(axes, ee_by_set):
        ax.plot(times, 1e-6 * pfe_by_set[ns], color=SERIES[1], label=f"PFE {quantile:.0%}")
        ax.plot(times, 1e-6 * ee_by_set[ns], color=SERIES[0], linewidth=2.2, label="Expected Exposure")
        ax.set_title(f"Netted exposure — {ns}")
        ax.set_xlabel("Time (years)")
        ax.set_xlim(times[0], times[-1])
    axes[0].set_ylim(bottom=0)  # shared axis: set once, after every panel has been drawn
    axes[0].set_ylabel("Exposure ($m)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_gross_vs_netted(times: np.ndarray, netted_ee: dict[str, np.ndarray], gross_ee: dict[str, np.ndarray]) -> plt.Figure:
    """One panel per netting set: gross EE (no netting) vs netted EE; the gap is the netting benefit."""
    apply_style()
    fig, axes = _panels(len(netted_ee))
    for ax, ns in zip(axes, netted_ee):
        gross, net = 1e-6 * gross_ee[ns], 1e-6 * netted_ee[ns]
        ax.fill_between(times, net, gross, color=SERIES[2], alpha=0.15, linewidth=0, label="Netting benefit")
        ax.plot(times, gross, color=SERIES[2], label="Gross EE (sum of trade exposures)")
        ax.plot(times, net, color=SERIES[0], linewidth=2.2, label="Netted EE")
        ax.set_title(f"Gross vs netted EE — {ns}")
        ax.set_xlabel("Time (years)")
        ax.set_xlim(times[0], times[-1])
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Expected Exposure ($m)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_exposure_distributions(times: np.ndarray, mtm_by_set: dict[str, np.ndarray], dates: tuple[float, ...],
                                quantile: float = 0.95, n_bins: int = 50) -> plt.Figure:
    """Rows = netting sets, columns = dates: histogram of netted value V(t).

    Bars with V > 0 (the exposure) are colored, bars with V <= 0 are neutral; the
    vertical lines mark EE and PFE of max(V, 0), computed on all paths. The x-range
    shows the central 99.6% of each netting set's values.
    """
    apply_style()
    n_rows, n_cols = len(mtm_by_set), len(dates)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.9 * n_cols, 2.9 * n_rows), squeeze=False)
    for row, (ns, mtm) in enumerate(mtm_by_set.items()):
        values = 1e-6 * mtm
        lo, hi = np.percentile(values, [0.2, 99.8])
        width = (hi - lo) / n_bins
        edges = np.arange(np.floor(lo / width), np.ceil(hi / width) + 1) * width  # 0 is always a bin edge
        for col, t in enumerate(dates):
            ax = axes[row, col]
            v = values[:, int(np.argmin(np.abs(times - t)))]
            exposure = np.maximum(v, 0.0)
            ax.hist(v[v <= 0], bins=edges, color=NEUTRAL_FILL, edgecolor=SURFACE, linewidth=0.4,
                    label="V ≤ 0 (no exposure)")
            ax.hist(v[v > 0], bins=edges, color=SERIES[0], alpha=0.45, edgecolor=SURFACE, linewidth=0.4,
                    label="V > 0 (exposure)")
            ax.axvline(exposure.mean(), color=SERIES[0], linewidth=2.2, label="EE")
            ax.axvline(np.quantile(exposure, quantile, method="inverted_cdf"), color=SERIES[1], linewidth=1.6,
                       linestyle="--", label=f"PFE {quantile:.0%}")
            ax.text(0.02, 0.95, f"P(E = 0) = {(exposure == 0).mean():.1%}", transform=ax.transAxes,
                    va="top", fontsize=8.5, color=TEXT_SECONDARY)
            ax.set_title(f"{ns} at t = {t:g}Y", fontsize=10.5)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            if row == n_rows - 1:
                ax.set_xlabel("Netted value to us ($m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig



# Credit and CVA charts color by counterparty / netting set: NS_A = SERIES[0], NS_B = SERIES[1],
# in the order the netting sets are passed. Netted vs gross keeps SERIES[0] / SERIES[2].


def plot_credit_curves(credits: dict[str, CounterpartyCredit], horizon: float = 10.0) -> plt.Figure:
    """Cumulative default probability (left) and default probability per year (right) for each counterparty."""
    apply_style()
    fig, (ax_cum, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.0))
    fine = np.linspace(0.0, horizon, 401)
    years = np.arange(0.0, horizon + 1.0)
    width = 0.8 / len(credits)
    for k, (color, credit) in enumerate(zip(SERIES, credits.values())):
        curve = credit.hazard_curve()
        label = f"{credit.counterparty_id} (assumed {credit.spread * 1e4:.0f}bp, R = {credit.recovery:.0%}, λ = {credit.hazard:.1%})"
        ax_cum.plot(fine, 100 * curve.cumulative_pd(fine), color=color, label=label)
        ax_bar.bar(years[1:] - 0.5 + (k - (len(credits) - 1) / 2) * width, 100 * curve.marginal_pd(years), width=width,
                   color=color, edgecolor=SURFACE, linewidth=0.8, label=credit.counterparty_id)
    ax_cum.set_title("Cumulative default probability")
    ax_cum.set_xlabel("Time (years)")
    ax_cum.set_ylabel("PD(t) (%)")
    ax_cum.set_xlim(0, horizon)
    ax_cum.set_ylim(bottom=0)
    ax_cum.legend(loc="upper left")
    ax_bar.set_title("Default probability in each year")
    ax_bar.set_xlabel("Year")
    ax_bar.set_ylabel("ΔPD (%)")
    ax_bar.set_xticks(years[1:] - 0.5, [f"{int(y)}" for y in years[1:]])
    ax_bar.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_cva_contributions(edges: np.ndarray, netted_by_set: dict[str, np.ndarray],
                           gross_by_set: dict[str, np.ndarray]) -> plt.Figure:
    """One panel per netting set: CVA contributed by each yearly bucket, netted vs gross, in $k."""
    apply_style()
    fig, axes = _panels(len(netted_by_set))
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = 0.38 * (edges[1] - edges[0])
    for ax, ns in zip(axes, netted_by_set):
        ax.bar(centers - width / 2, 1e-3 * gross_by_set[ns], width=width, color=SERIES[2], edgecolor=SURFACE,
               linewidth=0.8, label="Gross (no netting)")
        ax.bar(centers + width / 2, 1e-3 * netted_by_set[ns], width=width, color=SERIES[0], edgecolor=SURFACE,
               linewidth=0.8, label="Netted")
        ax.set_title(f"CVA by year — {ns}")
        ax.set_xlabel("Default year")
        ax.set_xticks(centers, [f"{int(e) + 1}" for e in edges[:-1]])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CVA contribution ($k)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_credit_sensitivity(table: pd.DataFrame) -> plt.Figure:
    """CVA vs assumed spread (left) and vs recovery (right), holding the simulated exposures fixed."""
    apply_style()
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(11, 4.0))
    for color, (ns, rows) in zip(SERIES, table.groupby("netting_set_id", sort=False)):
        spread = rows[rows["view"] == "spread"]
        ax_s.plot(spread["input"], 1e-3 * spread["cva"], color=color, marker="o", markersize=5, label=ns)
        fixed_h = rows[rows["view"] == "recovery_hazard_fixed"]
        fixed_s = rows[rows["view"] == "recovery_spread_fixed"]
        ax_r.plot(100 * fixed_h["input"], 1e-3 * fixed_h["cva"], color=color, marker="o", markersize=5,
                  label=f"{ns}, hazard fixed")
        ax_r.plot(100 * fixed_s["input"], 1e-3 * fixed_s["cva"], color=color, marker="s", markersize=5,
                  linestyle="--", label=f"{ns}, spread fixed")
    ax_s.set_title("CVA vs assumed counterparty spread")
    ax_s.set_xlabel("CDS-equivalent spread assumption (bp), recovery 40%")
    ax_s.set_ylabel("CVA ($k)")
    ax_s.set_ylim(bottom=0)
    ax_s.legend(loc="upper left")
    ax_r.set_title("CVA vs recovery rate")
    ax_r.set_xlabel("Recovery (%)")
    ax_r.set_ylabel("CVA ($k)")
    ax_r.set_ylim(bottom=0)
    ax_r.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# Collateral charts color by CSA scenario in the order given (categorical slots 0, 1, 2, 3):
# uncollateralized keeps SERIES[0], the netted-EE color used throughout. The gross
# reference bar is neutral grey.


def _escape_dollars(text: str) -> str:
    """Stop matplotlib from reading '$...$' in labels as mathtext."""
    return text.replace("$", r"\$")


def _short_stage_label(stage: str) -> str:
    """Compact x-axis label for a mitigation stage."""
    if stage.startswith("Gross"):
        return "Gross"
    if stage.startswith("Netted, uncollateralized"):
        return "Netted"
    if "Zero threshold" in stage:
        return "CSA\nH = 0"
    if "Threshold " in stage:
        return "CSA\nH " + _escape_dollars(stage.split("Threshold ")[1].split(",")[0])
    return _escape_dollars(stage)


def plot_collateral_profiles(profiles: pd.DataFrame) -> plt.Figure:
    """One panel per netting set: netted EE through time under each CSA scenario, in $m."""
    apply_style()
    sets = list(dict.fromkeys(profiles["netting_set_id"]))
    scenarios = list(dict.fromkeys(profiles["scenario"]))
    fig, axes = _panels(len(sets))
    for ax, ns in zip(axes, sets):
        for color, scenario in zip(SERIES, scenarios):
            rows = profiles[(profiles["netting_set_id"] == ns) & (profiles["scenario"] == scenario)]
            ax.plot(rows["time"], 1e-6 * rows["ee"], color=color, linewidth=2.0 if color == SERIES[0] else 1.6,
                    label=_escape_dollars(scenario))
        ax.set_title(f"EE under collateral — {ns}")
        ax.set_xlabel("Time (years)")
        ax.set_xlim(rows["time"].min(), rows["time"].max())
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Expected Exposure ($m)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_cva_mitigation(mitigation: pd.DataFrame) -> plt.Figure:
    """One panel per netting set: CVA from gross, to netted, to netted plus each CSA, in $k."""
    apply_style()
    sets = list(dict.fromkeys(mitigation["netting_set_id"]))
    fig, axes = _panels(len(sets), height=4.4)
    for ax, ns in zip(axes, sets):
        rows = mitigation[mitigation["netting_set_id"] == ns].reset_index(drop=True)
        colors = [NEUTRAL_FILL] + SERIES[: len(rows) - 1]
        positions = np.arange(len(rows))
        bars = ax.bar(positions, 1e-3 * rows["cva"], color=colors, edgecolor=SURFACE, linewidth=0.8, width=0.7)
        for bar, value in zip(bars, rows["cva"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value / 1e3:,.1f}", ha="center",
                    va="bottom", fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_xticks(positions, [_short_stage_label(stage) for stage in rows["stage"]], fontsize=8.5)
        ax.set_title(f"CVA after netting and collateral — {ns}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CVA ($k)")
    fig.tight_layout()
    return fig


def plot_convergence(table: pd.DataFrame) -> plt.Figure:
    """CVA estimate with 95% intervals vs number of paths per netting set, and the standard error vs N (log-log)."""
    apply_style()
    sets = [ns for ns in dict.fromkeys(table["netting_set_id"]) if ns != "TOTAL"]
    fig, axes = plt.subplots(1, len(sets) + 1, figsize=(5.0 * (len(sets) + 1), 4.0))
    for ax, color, ns in zip(axes, SERIES, sets):
        rows = table[table["netting_set_id"] == ns]
        n, cva = rows["n_paths"].to_numpy(), 1e-3 * rows["cva"].to_numpy()
        ax.fill_between(n, 1e-3 * rows["ci95_low"], 1e-3 * rows["ci95_high"], color=color, alpha=0.15, linewidth=0,
                        label="95% interval")
        ax.plot(n, cva, color=color, marker="o", markersize=5, label="CVA estimate")
        ax.axhline(cva[-1], color=TEXT_SECONDARY, linewidth=0.9, linestyle="--", label=f"{n[-1]:,} paths")
        ax.set_xscale("log")
        ax.set_title(f"CVA convergence — {ns}")
        ax.set_xlabel("Number of paths")
        ax.set_ylabel("CVA ($k)")
        ax.legend(loc="upper right", fontsize=8.5)
    ax = axes[-1]
    for color, ns in zip(SERIES, sets):
        rows = table[table["netting_set_id"] == ns]
        ax.plot(rows["n_paths"], 1e-3 * rows["cva_se"], color=color, marker="o", markersize=5, label=ns)
    first = table[table["netting_set_id"] == sets[-1]].iloc[0]
    n_ref = table["n_paths"].unique()
    ax.plot(n_ref, 1e-3 * first["cva_se"] * np.sqrt(first["n_paths"] / n_ref), color=TEXT_SECONDARY, linewidth=1.0,
            linestyle="--", label="∝ 1/√N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Standard error of CVA")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Std error ($k)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


SENSITIVITY_LABELS = {  # parameter: (panel title, x-axis label, display scale)
    "sigma": ("CVA vs rate volatility", "Hull-White σ (%/√yr)", 100.0),
    "mean_reversion": ("CVA vs mean reversion", "Mean reversion a (%/yr)", 100.0),
    "maturity_scale": ("CVA vs portfolio tenor", "Tenor multiplier (×)", 1.0),
}


def plot_market_sensitivity(table: pd.DataFrame) -> plt.Figure:
    """CVA vs rate volatility, mean reversion and portfolio tenor, one line per netting set, in $k."""
    apply_style()
    sets = [ns for ns in dict.fromkeys(table["netting_set_id"]) if ns != "TOTAL"]
    fig, axes = plt.subplots(1, len(SENSITIVITY_LABELS), figsize=(14, 4.0))
    for ax, (parameter, (title, label, scale)) in zip(axes, SENSITIVITY_LABELS.items()):
        for color, ns in zip(SERIES, sets):
            rows = table[(table["parameter"] == parameter) & (table["netting_set_id"] == ns)]
            ax.plot(scale * rows["value"], 1e-3 * rows["cva"], color=color, marker="o", markersize=5, label=ns)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel("CVA ($k)")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


TORNADO_NAMES = {
    "spread_scale": "Assumed spread ×{:g}",
    "recovery_hazard_fixed": "Recovery {:.0%} (hazard fixed)",
    "sigma": "σ = {:.2%}",
    "mean_reversion": "a = {:.0%}",
    "maturity_scale": "Tenor ×{:g}",
    "notional_scale": "Notional ×{:g}",
}


def plot_sensitivity_tornado(table: pd.DataFrame) -> plt.Figure:
    """Portfolio CVA change vs base for a low and high value of each input, widest range on top."""
    apply_style()
    spans = table.groupby("parameter")["change_vs_base"].agg(lambda v: v.max() - v.min()).sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(spans) + 1.2))
    for row_pos, parameter in enumerate(spans.index):
        rows = table[table["parameter"] == parameter].sort_values("value")
        for color, (_, row) in zip((SERIES[0], SERIES[1]), rows.iterrows()):
            change = 100 * row["change_vs_base"]
            ax.barh(row_pos, change, color=color, height=0.6, edgecolor=SURFACE, linewidth=0.8)
            ax.text(change + (1.2 if change >= 0 else -1.2), row_pos,
                    _escape_dollars(TORNADO_NAMES[parameter].format(row["value"])) + f"  ({change:+.0f}%)",
                    va="center", ha="left" if change >= 0 else "right", fontsize=8.5, color=TEXT_PRIMARY)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
    ax.set_yticks([])
    limit = 100 * table["change_vs_base"].abs().max() * 1.9
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(f"Change in portfolio CVA vs base (${table['base_cva'].iloc[0] / 1e3:,.0f}k)".replace("$", r"\$"))
    ax.set_title("What moves portfolio CVA")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    ax.legend(handles, ["Lower input", "Higher input"], loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig
