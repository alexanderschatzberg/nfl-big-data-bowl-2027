"""One figure showing the braking measurement on a drill trace and on a game trace.

Two receivers from the headline sample, each drawn once in the Combine comeback
drill and once on an NFL hitch route, with the half second of hardest braking
marked. The values printed are `attempt_features(...)["dec_win05"]` for exactly
the clips drawn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import matched
from .data import connect
from .figures import BLUE, ORANGE, _title, plt
from .kinematics import _smooth, attempt_features

FEATURE, LAG = "dec_win05", 5
DRILL = "COMEBACK_ROUTE_RIGHT"
# Two 2023-class receivers from the headline sample with a complete comeback clip and plenty of hitch
# routes: Mims ranks 6th of 54 in drill braking and 9th in games, Nacua 38th and 44th. Same class
# because drill setups change by year.
PLAYERS = {55928: "hard", 56042: "soft"}
SHOWN = 5.5  # seconds drawn; game plays run on after the route


def _trace(g: pd.DataFrame) -> dict:
    """Smoothed speed as the feature sees it, the feature value, and where its window sits."""
    value = attempt_features(g["t"].values, g["x"].values, g["y"].values, g["s"].values)[FEATURE]
    t = g["t"].to_numpy(float) - g["t"].iloc[0]
    s = np.clip(_smooth(g["s"].to_numpy(float)), 0, None)
    i = int(np.argmin((s[LAG:] - s[:-LAG]) / (t[LAG:] - t[:-LAG])))
    assert np.isclose((s[i] - s[i + LAG]) / (t[i + LAG] - t[i]), value)
    return {"t": t, "s": s, "window": (i, i + LAG), "value": float(value)}


def _drill(con, nfl_id: int) -> dict:
    """The comeback attempt closest to the player's median over attempts (the value the analysis uses)."""
    df = con.execute(
        f"""
        SELECT event_id, attempt, epoch(time) AS t, x, y, s FROM combine_tracking
        WHERE entity_type = 'PLAYER' AND nfl_id = {nfl_id} AND drill_name = '{DRILL}'
        ORDER BY event_id, attempt, time
        """
    ).df()
    clips = [{"event_id": k[0], "attempt": int(k[1])} | _trace(g) for k, g in df.groupby(["event_id", "attempt"])]
    median = np.median([c["value"] for c in clips])
    return min(clips, key=lambda c: abs(c["value"] - median))


def _game(con, nfl_id: int, plays: pd.DataFrame, pair: matched.Pair) -> dict:
    """The hitch route closest to the player's 90th percentile, traced from the snap."""
    h = plays[(plays["nfl_id"] == nfl_id) & (plays["position"] == pair.position)].query(pair.plays)
    target = h[FEATURE].quantile(pair.quantile)
    p = h.loc[(h[FEATURE] - target).abs().idxmin()]
    g = con.execute(
        f"""
        WITH snap AS (SELECT min(time) AS ts FROM game_tracking
                      WHERE game_id = {p['game_id']} AND play_id = {p['play_id']} AND nfl_id = {nfl_id}
                        AND event = 'ball_snap')
        SELECT epoch(time) AS t, x, y, s FROM game_tracking, snap
        WHERE game_id = {p['game_id']} AND play_id = {p['play_id']} AND nfl_id = {nfl_id} AND time >= ts
        ORDER BY time
        """
    ).df()
    out = {"game_id": int(p["game_id"]), "play_id": int(p["play_id"]), "season": int(p["season"]),
           "p90": float(target), "hitches": len(h)} | _trace(g)
    assert np.isclose(out["value"], p[FEATURE])
    return out


def data() -> dict:
    """Per receiver: name, and the drill and game clips with the braking window and value."""
    con = connect()
    pair = next(p for p in matched.PAIRS if p.name == "stop-route braking")
    plays = matched.game_plays()
    sample = matched.pair_frame(pair, plays).set_index("nfl_id")
    names = con.execute("SELECT nfl_id, display_name FROM players").df().set_index("nfl_id")["display_name"]
    out = {}
    for nfl_id, kind in PLAYERS.items():
        game = _game(con, nfl_id, plays, pair)
        assert np.isclose(game["p90"], sample.loc[nfl_id, "g"])
        out[kind] = {"nfl_id": nfl_id, "name": names[nfl_id], "drill": _drill(con, nfl_id), "game": game}
    return out


def figure(d: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.7), sharey=True)
    panels = {"drill": ("Combine comeback drill", "Seconds from the start of the run"),
              "game": ("NFL hitch route", "Seconds from the snap")}
    for ax, (side, (title, xlabel)) in zip(axes, panels.items()):
        for kind, color in (("hard", BLUE), ("soft", ORANGE)):
            c = d[kind][side]
            keep = c["t"] <= SHOWN
            i, j = c["window"]
            ax.plot(c["t"][keep], c["s"][keep], color=color, lw=1.8, zorder=3)
            ax.plot(c["t"][i:j + 1], c["s"][i:j + 1], color=color, lw=7, solid_capstyle="butt", zorder=4)
            # Hard braker labelled under the speed hump, soft braker in the empty top corner.
            x, y, ha = (1.55, 2.6, "center") if kind == "hard" else (SHOWN, 9.5, "right")
            ax.text(x, y, f"{d[kind]['name']}\n{c['value']:.1f}", color=color, fontsize=11.5, fontweight="bold",
                    va="top", ha=ha, linespacing=1.15)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xlim(0, SHOWN)
        ax.set_ylim(0, 9.6)
        ax.tick_params(labelsize=10.5)
    axes[0].set_ylabel("Speed (yards per second)", fontsize=11)
    _title(fig, "Braking, measured the same way in a drill and in a game",
           "The thick stretch is the half second in which each receiver loses the most speed. "
           "The number is that braking in yards per second lost per second.", top=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return fig


def draw(path: Path = Path("report/figures/fig6_traces.png")) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(data())
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
