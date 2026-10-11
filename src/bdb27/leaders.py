"""Who the headline is about: the receivers in the stop-route braking sample, by name.

One row per receiver with his drill braking, his game braking on hitch routes
and where each ranks. Rank 1 brakes hardest; ties go to the lower `nfl_id`.
Separation is `value._outcome`: cushion-adjusted, missing below its play minimum.
"""
import numpy as np
import pandas as pd

from . import matched, value
from .data import connect

HEADLINE = "stop-route braking"


def _rank(s: pd.Series, ids: pd.Series) -> pd.Series:
    order = np.lexsort((ids.to_numpy(), -s.to_numpy()))
    return pd.Series(np.arange(1, len(s) + 1), index=s.index[order]).reindex(s.index)


def table() -> pd.DataFrame:
    """The headline sample, hardest drill braker first."""
    pair = next(p for p in matched.PAIRS if p.name == HEADLINE)
    plays = matched.game_plays()
    d = matched.pair_frame(pair, plays)[["nfl_id", "draft_year", "forty", "c", "g"]]
    d = d.rename(columns={"c": "drill_z", "g": "game_braking"}).sort_values("nfl_id").reset_index(drop=True)
    sub = plays[plays["position"] == pair.position].query(pair.plays)
    hitch = sub.groupby("nfl_id")[pair.feature].count().rename("hitch_routes")
    routes = plays[plays["route_ran"].notna()].groupby("nfl_id").size().rename("routes")
    # `hitches` has no ORDER BY; unsorted rows move the cushion fit in the last bits.
    h = value.hitches(pair.position).sort_values(["game_id", "play_id", "nfl_id"]).reset_index(drop=True)
    sep = value._outcome(h, "separation", None).rename("hitch_separation")
    sep_man = value._outcome(h, "separation", "MAN_COVERAGE").rename("hitch_separation_man")
    who = connect().execute(
        """
        SELECT nfl_id, display_name AS name, draft_overall_pick AS pick, career_offensive_snaps AS snaps
        FROM players JOIN player_career_successes USING (nfl_id)
        """
    ).df()
    for s in (hitch, routes, sep, sep_man):
        d = d.merge(s, on="nfl_id", how="left")
    d = d.merge(who, on="nfl_id", how="left")
    d["pick"] = d["pick"].astype("Int64")
    d["drill_rank"] = _rank(d["drill_z"], d["nfl_id"])
    d["game_rank"] = _rank(d["game_braking"], d["nfl_id"])
    cols = ["nfl_id", "name", "draft_year", "pick", "drill_z", "drill_rank", "game_braking", "game_rank",
            "hitch_routes", "hitch_separation", "hitch_separation_man", "routes", "snaps", "forty"]
    return d[cols].sort_values("drill_rank").reset_index(drop=True)


def top_bottom(n: int = 5) -> pd.DataFrame:
    """The `n` hardest and `n` softest drill brakers, with headers for the writeup.

    `attrs` counts how many of each group sit in the top half of the game ranking.
    """
    t = table()
    half = len(t) // 2
    ends = pd.concat([t.head(n), t.tail(n)])
    pick = ends["pick"].astype(object).where(ends["pick"].notna(), "undrafted")
    out = pd.DataFrame({
        "Receiver": ends["name"],
        "Draft": ends["draft_year"].astype(str) + ", " + np.where(ends["pick"].notna(), "pick ", "") + pick.astype(str),
        "Drill braking rank": ends["drill_rank"],
        "Hardest stops on game hitch routes (yd/s²)": ends["game_braking"].round(1),
        f"Game braking rank (of {len(t)})": ends["game_rank"],
    }).reset_index(drop=True)
    out.attrs = {"n": len(t), "top_in_game_top_half": int((t.head(n)["game_rank"] <= half).sum()),
                 "bottom_in_game_top_half": int((t.tail(n)["game_rank"] <= half).sum())}
    return out
