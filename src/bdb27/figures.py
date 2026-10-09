"""The writeup's figures. Each function takes the table it draws and returns a matplotlib Figure."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import matched, report

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
CLASS_COLOR = {2023: BLUE, 2024: ORANGE, 2025: AQUA}
CLASS_MARKER = {2023: "o", 2024: "s", 2025: "^"}  # shape as well as colour

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _title(fig, title: str, subtitle: str, top: float = 0.98) -> None:
    import textwrap

    fig.text(0.012, top, title, fontsize=13, fontweight="bold", va="top", ha="left")
    subtitle = "\n".join(textwrap.wrap(subtitle, int(fig.get_figwidth() * 13.5)))
    fig.text(0.012, top - 0.062 * 4.2 / fig.get_figheight(), subtitle, fontsize=9.5, color=MUTED, va="top", ha="left",
             linespacing=1.35)


def _fit(ax, x, y, color=MUTED) -> None:
    b = np.polyfit(x, y, 1)
    xs = np.array([np.min(x), np.max(x)])
    ax.plot(xs, np.polyval(b, xs), color=color, lw=1.5, zorder=2)


def stopwatch(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.9))
    units = {"game_speed_p99": "yd/s", "sep_oe_all": "yd", "targets_per_route": "per route", "yards_per_route": "yd per route"}
    for ax, (col, name) in zip(axes, report.STOPWATCH_OUTCOMES.items()):
        d = frame.dropna(subset=["forty", col])
        rho, p = stats.spearmanr(d["forty"], d[col])
        ax.scatter(d["forty"], d[col], s=26, color=BLUE if p < 0.05 else GREY, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        _fit(ax, d["forty"], d[col])
        ax.set_title(name)
        ax.set_xlabel("40-yard dash (s)")
        ax.set_ylabel(f"vs alignment average ({units[col]})")
        ax.text(0.97, 0.96, f"ρ = {rho:+.2f}\nn = {len(d)}".replace("-", "−"), transform=ax.transAxes, ha="right", va="top", fontsize=10,
                fontweight="bold" if p < 0.05 else "normal")
    _title(fig, "The 40 predicts how fast a receiver plays, and little else",
           "Wide receivers, 2023–25 draft classes. Each dot is a player; outcomes are regular-season values relative to "
           "slot or outside receivers. ρ is the rank correlation.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


def transfer_scatter(d: pd.DataFrame, numbers: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for year, g in d.groupby("draft_year"):
        rho = stats.spearmanr(g["c"], g["g"])[0]
        ax.scatter(g["c"], g["g"], s=46, color=CLASS_COLOR[year], marker=CLASS_MARKER[year], edgecolor=SURFACE,
                   linewidth=1, zorder=3, label=f"{year} class, ρ = {rho:.2f}")
    _fit(ax, d["c"], d["g"], INK)
    ax.set_xlabel("Braking in the Combine comeback and curl drills\n(standard deviations from the draft-class average)")
    ax.set_ylabel("Braking on hitch routes in NFL games\n(yd/s², the player's 90th-percentile play)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, handletextpad=0.2, columnspacing=1.6,
              borderaxespad=0.2)
    h = numbers[report.HEADLINE]
    ax.text(0.03, 0.97, f"ρ = {h['rho']:.2f}\n{h['partial_rho']:.2f} after removing 40 time,\nsize and alignment",
            transform=ax.transAxes, va="top", fontsize=10)
    _title(fig, "Receivers who brake hard in Indianapolis brake hard on Sundays",
           f"{h['n']} wide receivers with both drills and at least 20 regular-season hitch routes.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def benchmark(benchmarks: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    labels = {report.HEADLINE: "Braking on hitch routes", report.SPEED: "Top speed on go routes"}
    y, ticks = 0.0, []
    for name, label in labels.items():
        b = benchmarks[name]
        ax.text(0, y, label, fontweight="bold", va="center", ha="left")
        y += 0.62
        rows = [("Combine drill", b["combine_to_game"], b["combine_ci"], BLUE, f"{b['n']} players"),
                ("Previous NFL season", b["season_to_next_season"], b["season_ci"], GREY,
                 f"{b['season_pairs']} season pairs")]
        for who, v, ci, color, n in rows:
            ax.barh(y, v, height=0.5, color=color, zorder=3)
            ax.plot(ci, [y, y], color=INK, lw=1.2, zorder=4)
            ax.text(0.9, y, f"{v:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")
            ax.text(1.0, y, n, va="center", ha="left", fontsize=9, color=MUTED)
            ticks.append((y, who))
            y += 0.62
        y += 0.35
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y - 0.6, -0.4)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Rank correlation with game movement (line = 95% bootstrap interval)", loc="left")
    ax.grid(axis="y", visible=False)
    _title(fig, "A Combine drill tells you nearly as much as a season of NFL tracking",
           "How well each source predicts a receiver's in-game movement capability: his Combine drill, or the same "
           "game measure from his own previous season.", top=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    return fig


ROBUST_LABELS = {
    "as specified": ("Main result", "As specified (90th-percentile play, ≥20 hitches)"),
    "game quantile 0.5": ("Which game plays count", "Median play"),
    "game quantile 0.75": ("Which game plays count", "75th-percentile play"),
    "game quantile 0.85": ("Which game plays count", "85th-percentile play"),
    "game quantile 0.95": ("Which game plays count", "95th-percentile play"),
    "min plays 10": ("Minimum hitch routes", "At least 10"),
    "min plays 30": ("Minimum hitch routes", "At least 30"),
    "min plays 40": ("Minimum hitch routes", "At least 40"),
    "feature dec_win10": ("How braking is measured", "Sustained over 1.0 s"),
    "feature dec_p95": ("How braking is measured", "95th-percentile deceleration"),
    "feature dec_max": ("How braking is measured", "Peak deceleration"),
    "drill COMEBACK_ROUTE_RIGHT only": ("Which drill", "Comeback drill only"),
    "drill CURL_ROUTE_RIGHT only": ("Which drill", "Curl drill only"),
    "all breaking routes": ("Which game routes", "All breaking routes"),
    "placebo: go routes in games": ("Placebos (should be zero)", "Stop drills → braking on go routes"),
    "placebo: go-route drill": ("Placebos (should be zero)", "Go-route drill → hitch braking"),
    "placebo: 40-yard dash": ("Placebos (should be zero)", "40-yard dash → hitch braking"),
}


def robustness(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    lo, hi = table.attrs["ci95"]
    ax.axvspan(lo, hi, color=BLUE, alpha=0.10, zorder=1)
    ax.axvline(0, color=INK, lw=1, zorder=2)
    y, ticks, last = 0.0, [], None
    for _, r in table.iterrows():
        group, label = ROBUST_LABELS[r["variant"]]
        if group != last:
            y += 0.55
            ax.text(-0.985, y, group, fontweight="bold", va="center", ha="left", fontsize=9.5)
            y += 0.85
            last = group
        placebo = group.startswith("Placebo")
        ax.scatter(r["partial_rho"], y, s=62, color=ORANGE if placebo else BLUE, marker="D" if placebo else "o",
                   edgecolor=SURFACE, linewidth=1, zorder=4)
        ax.text(0.80, y, f"{r['partial_rho']:+.2f}".replace("-", "−"), va="center", ha="right", fontsize=9)
        ticks.append((y, label))
        y += 0.8
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y, -0.2)
    ax.set_xlim(-0.45, 0.82)
    ax.set_xlabel("Drill-to-game braking correlation\n(net of 40 time, size and alignment)")
    ax.grid(axis="y", visible=False)
    for t in ax.texts[:]:
        if t.get_position()[0] == -0.985:
            t.set_transform(ax.get_yaxis_transform())
            t.set_x(-0.62)
    _title(fig, "The braking result holds however it is cut, and the placebos stay at zero",
           "Shaded band: 95% bootstrap interval of the main result. Blue circles are variations of the test; "
           "orange diamonds pair drills and plays that should not match.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return fig


def chain(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axvline(0, color=INK, lw=1, zorder=2)
    y, ticks, last = 0.0, [], None
    for _, r in table.iterrows():
        if r["stage"] != last:
            y += 0.55
            ax.text(-0.72, y, r["stage"], fontweight="bold", va="center", ha="left", fontsize=9.5,
                    transform=ax.get_yaxis_transform())
            y += 0.85
            last = r["stage"]
        solid = r["lo"] > 0 or r["hi"] < 0
        color = BLUE if r["stage"].startswith("Transfer") else GREY
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=2, zorder=3, solid_capstyle="round")
        ax.scatter(r["partial_rho"], y, s=64, color=color if solid else SURFACE, edgecolor=color, linewidth=2, zorder=4)
        label = r["label"].split(" → ")[-1] if not r["stage"].startswith("Transfer") else r["label"]
        label = label[0].upper() + label[1:] + ("  (pre-specified test)" if r["pre_specified"] else "")
        ax.text(0.98, y, f"{r['partial_rho']:+.2f}".replace("-", "−"), va="center", ha="right", fontsize=9.5, fontweight="bold")
        ax.text(1.04, y, f"n = {r['n']}", va="center", ha="left", fontsize=9, color=MUTED)
        ticks.append((y, label))
        y += 0.8
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y, -0.2)
    ax.set_xlim(-0.75, 1.3)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("Correlation net of 40 time, size and alignment\n(line = 95% bootstrap interval)", loc="left")
    ax.grid(axis="y", visible=False)
    _title(fig, "The movement transfers. It does not buy separation.",
           "Wide receivers. Filled dots: interval excludes zero. Braking predicts braking, but neither drill braking "
           "nor game braking predicts results on hitch routes.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return fig


def card(d: pd.DataFrame, numbers: dict) -> plt.Figure:
    """560 x 280 thumbnail."""
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    ax = fig.add_axes((0.56, 0.12, 0.41, 0.78))
    for year, g in d.groupby("draft_year"):
        ax.scatter(g["c"], g["g"], s=20, color=CLASS_COLOR[year], marker=CLASS_MARKER[year], edgecolor=SURFACE, linewidth=0.6, zorder=3)
    _fit(ax, d["c"], d["g"], INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Combine drill braking", fontsize=8)
    ax.set_ylabel("Game braking", fontsize=8)
    fig.text(0.04, 0.86, "Brakes travel.", fontsize=19, fontweight="bold", va="top")
    fig.text(0.04, 0.62, "Separation\ndoesn't follow.", fontsize=15, va="top", color=MUTED)
    fig.text(0.04, 0.16, f"ρ = {numbers[report.HEADLINE]['rho']:.2f}  ·  {numbers[report.HEADLINE]['n']} receivers",
             fontsize=9, color=MUTED)
    return fig


def save_all(out: Path, *, stopwatch_frame, scatter, numbers, benchmarks, robustness_table, chain_table) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    figs = {"fig1_stopwatch": stopwatch(stopwatch_frame), "fig2_braking_transfer": transfer_scatter(scatter, numbers),
            "fig3_benchmark": benchmark(benchmarks), "fig4_robustness": robustness(robustness_table),
            "fig5_chain": chain(chain_table)}
    paths = []
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.png", dpi=200)
        paths.append(out / f"{name}.png")
    c = card(scatter, numbers)
    c.savefig(out / "card_560x280.png", dpi=100)
    plt.close("all")
    return paths + [out / "card_560x280.png"]


def build(out: Path = Path("report/figures"), n_boot: int = 2000) -> list[Path]:
    """Compute only what the figures need and write them (the `bdb27 figures` command)."""
    rb = matched.robustness(n_boot=n_boot)
    benchmarks = {n: matched.benchmark(n) for n in (report.HEADLINE, report.SPEED)}
    d = report.scatter()
    t = matched.test(d)
    numbers = {report.HEADLINE: {"n": t["n"], "rho": t["rho"], "partial_rho": t["partial_rho"]}}
    return save_all(out, stopwatch_frame=report.stopwatch_frame(), scatter=d, numbers=numbers, benchmarks=benchmarks,
                    robustness_table=rb, chain_table=report.chain(n_boot))
