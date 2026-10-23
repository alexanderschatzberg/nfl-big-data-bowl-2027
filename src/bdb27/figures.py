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


def _num(v: float) -> str:
    return f"{v:.2f}".replace("-", "−")


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _fit(ax, x, y, color=MUTED) -> None:
    b = np.polyfit(x, y, 1)
    xs = np.array([np.min(x), np.max(x)])
    ax.plot(xs, np.polyval(b, xs), color=color, lw=1.5, zorder=2)


def stopwatch(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 4.5))
    units = {"game_speed_p99": "yards per second", "sep_oe_all": "yards", "targets_per_route": "targets per route",
             "yards_per_route": "yards per route"}
    for ax, (col, name) in zip(axes, report.STOPWATCH_OUTCOMES.items()):
        d = frame.dropna(subset=["forty", col])
        rho, p = stats.spearmanr(d["forty"], d[col])
        ax.scatter(d["forty"], d[col], s=26, color=BLUE if p < 0.05 else GREY, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        _fit(ax, d["forty"], d[col])
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_title(name, pad=24)
        ax.text(0, 1.03, f"ρ = {_num(rho)}, {_p(p)}", transform=ax.transAxes, fontsize=9.5,
                color=INK if p < 0.05 else MUTED, fontweight="bold" if p < 0.05 else "normal")
        ax.set_xlabel("40-yard dash time (seconds)")
        ax.set_ylabel(f"Difference from average\n({units[col]})")
    _title(fig, "The 40-yard dash and four game outcomes for wide receivers",
           "65 wide receivers, 2023–2025 draft classes. Outcomes are regular-season values centred within alignment "
           "(slot or outside). Spearman ρ.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


LABELLED = ["Rashee Rice", "Puka Nacua", "Marvin Mims"]  # discussed in the text, alongside the five hardest and softest


def _label_receivers(ax, d: pd.DataFrame) -> None:
    """Name the five hardest and five softest drill brakers, and the receivers the text discusses."""
    from .data import connect

    names = connect().execute("SELECT nfl_id, display_name FROM players").df().set_index("nfl_id")["display_name"]
    d = d.assign(name=d["nfl_id"].map(names)).sort_values("c")
    pick = pd.concat([d.head(5), d.tail(5), d[d["name"].isin(LABELLED)]]).drop_duplicates("nfl_id")
    nudge = {"Troy Franklin": ((0, 9), "center"), "Quentin Johnston": ((6, -7), "left"),
             "Jalen McMillan": ((6, 5), "left")}  # keep neighbouring labels apart
    for _, r in pick.iterrows():
        right = r["c"] < d["c"].median()
        offset, ha = nudge.get(r["name"], ((6 if right else -6, 0), "left" if right else "right"))
        ax.annotate(r["name"], (r["c"], r["g"]), xytext=offset, textcoords="offset points",
                    ha=ha, va="center", fontsize=7.5, color=MUTED, zorder=5)


def transfer_scatter(d: pd.DataFrame, numbers: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for year, g in d.groupby("draft_year"):
        rho = stats.spearmanr(g["c"], g["g"])[0]
        ax.scatter(g["c"], g["g"], s=46, color=CLASS_COLOR[year], marker=CLASS_MARKER[year], edgecolor=SURFACE,
                   linewidth=1, zorder=3, label=f"{year} draft class (ρ = {_num(rho)})")
    _fit(ax, d["c"], d["g"], INK)
    _label_receivers(ax, d)
    ax.set_xlabel("Braking in the Combine comeback and curl drills\n(z-score within draft class)")
    ax.set_ylabel("Braking on hitch routes in NFL games\n(yards per second², his 90th-percentile play)")
    ax.set_ylim(top=d["g"].max() + 0.7)
    ax.legend(loc="upper right", fontsize=9, handletextpad=0.2, borderaxespad=0.6)
    h = numbers[report.HEADLINE]
    ax.text(0.03, 0.97, f"Spearman ρ = {_num(h['rho'])}, n = {h['n']}\n"
            f"Partial ρ = {_num(h['partial_rho'])}, {_p(h['partial_p'])}\n(controls: 40 time, height, weight, alignment)",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.4)
    _title(fig, "Braking in Combine drills and braking in NFL games",
           f"{h['n']} wide receivers who ran both drills and at least 20 regular-season hitch routes.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def benchmark(benchmarks: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    labels = {report.HEADLINE: "Predicting braking on hitch routes", report.SPEED: "Predicting top speed on go routes"}
    y, ticks = 0.0, []
    for name, label in labels.items():
        b = benchmarks[name]
        ax.text(0, y, label, fontweight="bold", va="center", ha="left")
        ax.text(0, y + 0.33, f"{b['season_pairs']} pairs of seasons from {b['season_players']} receivers",
                va="center", ha="left", fontsize=8.5, color=MUTED)
        y += 0.8
        rows = [("From his Combine drills", b["combine_to_next_season"], b["combine_next_ci"], BLUE),
                ("From his previous NFL season", b["season_to_next_season"], b["season_ci"], GREY)]
        for who, v, ci, color in rows:
            ax.barh(y, v, height=0.5, color=color, zorder=3)
            ax.plot(ci, [y, y], color=INK, lw=1.2, zorder=4)
            ax.text(0.9, y, _num(v), va="center", ha="left", fontsize=10, fontweight="bold")
            ticks.append((y, who))
            y += 0.62
        y += 0.35
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y - 0.6, -0.4)
    ax.set_xlim(-0.05, 1.0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Spearman ρ with the same measure in the next NFL season\n(line: 95% bootstrap CI, resampling receivers)", loc="left")
    ax.grid(axis="y", visible=False)
    _title(fig, "The Combine compared with a season of NFL tracking",
           "How well a receiver's movement in an NFL season is predicted by his Combine drills, and by the same "
           "measurement from his previous season.",
           top=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


ROBUST_LABELS = {
    "as specified": ("Main result", "As reported in the text"),
    "game quantile 0.5": ("Which of his hitch routes represents him", "His median play"),
    "game quantile 0.75": ("Which of his hitch routes represents him", "His 75th-percentile play"),
    "game quantile 0.85": ("Which of his hitch routes represents him", "His 85th-percentile play"),
    "game quantile 0.95": ("Which of his hitch routes represents him", "His 95th-percentile play"),
    "min plays 10": ("Minimum number of hitch routes", "At least 10"),
    "min plays 30": ("Minimum number of hitch routes", "At least 30"),
    "min plays 40": ("Minimum number of hitch routes", "At least 40"),
    "feature dec_win10": ("How braking is measured", "Averaged over 1 second"),
    "feature dec_p95": ("How braking is measured", "95th-percentile deceleration"),
    "feature dec_max": ("How braking is measured", "Peak deceleration"),
    "drill COMEBACK_ROUTE_RIGHT only": ("Which drill", "Comeback drill only"),
    "drill CURL_ROUTE_RIGHT only": ("Which drill", "Curl drill only"),
    "keeping cut-off drill clips": ("Which drill", "Both drills, keeping clips that were cut off early"),
    "all breaking routes": ("Which game routes", "All breaking routes"),
    "controlling for speed on hitch routes": ("Extra control", "His typical top speed on hitch routes"),
    "placebo: go routes in games": ("Placebos", "Comeback and curl drills, with go routes in games"),
    "placebo: go-route drill": ("Placebos", "Go-route drill, with hitch routes in games"),
    "placebo: 40-yard dash": ("Placebos", "40-yard dash, with hitch routes in games"),
}


def _rows(ax, table, group_of, draw, header_x: float) -> None:
    """Grouped rows down the page: a bold header per group, then `draw(row, y)` returning the tick label."""
    y, ticks, last = 0.0, [], None
    for _, r in table.iterrows():
        group = group_of(r)
        if group != last:
            y += 0.55
            ax.text(header_x, y, group, fontweight="bold", va="center", ha="right", fontsize=9.5,
                    transform=ax.get_yaxis_transform())
            y += 0.85
            last = group
        ticks.append((y, draw(r, y)))
        y += 0.8
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y, -0.2)
    ax.grid(axis="y", visible=False)


def robustness(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 6.9))
    ax.axvline(0, color=INK, lw=1, zorder=2)

    def draw(r, y):
        group, label = ROBUST_LABELS[r["variant"]]
        placebo = r["variant"].startswith("placebo")
        ax.scatter(r["partial_rho"], y, s=62, color=ORANGE if placebo else BLUE, marker="D" if placebo else "o",
                   edgecolor=SURFACE, linewidth=1, zorder=4)
        ax.text(0.66, y, _num(r["partial_rho"]), va="center", ha="right", fontsize=9)
        ax.text(0.70, y, _p(r["partial_p"]), va="center", ha="left", fontsize=9, color=MUTED)
        return label

    _rows(ax, table, lambda r: ROBUST_LABELS[r["variant"]][0], draw, header_x=-0.035)
    ax.set_xlim(-0.45, 0.95)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xlabel("Partial Spearman ρ, drill braking vs game braking\n(controls: 40 time, height, weight, alignment)", loc="left")
    _title(fig, "The braking result under different analysis choices",
           "Each row changes one analysis choice. Orange diamonds are placebos: a drill paired with a game situation it "
           "does not rehearse.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def chain(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    ax.axvline(0, color=INK, lw=1, zorder=2)

    def draw(r, y):
        transfer = r["stage"].startswith("Combine movement")
        solid = r["lo"] > 0 or r["hi"] < 0
        color = BLUE if transfer else GREY
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=2, zorder=3, solid_capstyle="round")
        ax.scatter(r["partial_rho"], y, s=64, color=color if solid else SURFACE, edgecolor=color, linewidth=2, zorder=4)
        ax.text(0.98, y, _num(r["partial_rho"]), va="center", ha="right", fontsize=9.5, fontweight="bold")
        ax.text(1.04, y, f"{r['n']} receivers", va="center", ha="left", fontsize=9, color=MUTED)
        label = r["label"] if transfer else r["label"].split(": ")[-1]
        return label[0].upper() + label[1:] + (" (specified in advance)" if r["pre_specified"] else "")

    _rows(ax, table, lambda r: r["stage"], draw, header_x=-0.02)
    ax.set_xlim(-0.75, 1.42)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("Partial Spearman ρ (controls: 40 time, height,\nweight, alignment); line: 95% bootstrap CI", loc="left")
    _title(fig, "Braking carries over to games but does not improve results on hitch routes",
           "Wide receivers. Filled dots: 95% CI excludes zero.", top=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    fig.text(0.04, 0.86, "Receiver braking", fontsize=19, fontweight="bold", va="top")
    fig.text(0.04, 0.62, "Combine drills and\nNFL games", fontsize=15, va="top", color=MUTED)
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
    from . import traces  # imports this module for its style

    paths.append(traces.draw(out / "fig6_traces.png"))
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
    numbers = {report.HEADLINE: {"n": t["n"], "rho": t["rho"], "partial_rho": t["partial_rho"], "partial_p": t["partial_p"]}}
    return save_all(out, stopwatch_frame=report.stopwatch_frame(), scatter=d, numbers=numbers, benchmarks=benchmarks,
                    robustness_table=rb, chain_table=report.chain(n_boot))
