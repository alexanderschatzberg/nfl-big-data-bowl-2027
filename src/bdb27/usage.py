"""Do teams deploy players according to their Combine movement?

The other modules ask whether a drill trait predicts how well a player does.
This asks whether it predicts what he is asked to do: the share of his routes
that are hitches or go routes, or the share of his snaps at one alignment.
Partial correlations remove 40 time, weight and height, and for route shares
also slot share, since slot and outside receivers run different route trees.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate, matched
from .data import artifacts_dir, connect

CUT_ROUTES = ("OUT", "IN", "SLANT", "CORNER", "POST")
MIN_PLAYS = 100


@dataclass(frozen=True)
class Use:
    position: str
    trait: str
    drills: list
    feature: str
    share: str  # column of `shares`


USES = [
    Use("WR", "stop-route braking", matched._STOP, "dec_win05", "route_HITCH"),
    Use("WR", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "route_GO"),
    Use("WR", "go-route drill top speed", ["GO_ROUTE_RIGHT"], "vmax", "route_GO"),
    Use("WR", "cut turn rate", matched._CUT, "cut_turn_rate", "align_SLOT_WR"),
    Use("WR", "cut lateral acceleration", matched._CUT, "lat_acc", "route_CUT"),
    Use("TE", "route braking", ["HOOK_ROUTE_RIGHT", "IN_ROUTE_LEFT"], "dec_win05", "route_HITCH"),
    Use("TE", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "route_GO"),
    Use("TE", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "align_SLOT_WR"),
    Use("DL", "hoop curve speed", ["RUN_THE_HOOP_DRILL"], "curve_speed", "align_EDGE"),
    Use("DL", "hoop lateral acceleration", ["RUN_THE_HOOP_DRILL"], "lat_acc", "align_EDGE"),
    Use("DL", "pass-rush drill burst", ["PASS_RUSH_DRILL"], "acc_win05", "align_EDGE"),
    Use("OL", "mirror lateral acceleration", matched._MIRROR, "lat_acc", "align_T"),
    Use("OL", "mirror braking", matched._MIRROR, "dec_win05", "align_T"),
    Use("OL", "pull top speed", matched._PULL, "vmax", "align_T"),
    Use("DB", "break-drill turn rate", matched._DB_BREAK, "cut_turn_rate", "align_SLOT_CB"),
    Use("DB", "break-drill braking", matched._DB_BREAK, "dec_win05", "align_HIGH_SAFETY"),
    Use("DB", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "align_CB"),
]


def shares() -> pd.DataFrame:
    """One row per player: share of routes by type and share of snaps by alignment."""
    p = connect().execute(
        """
        SELECT pp.nfl_id, pp.lined_up_position AS align, pp.route_ran AS route
        FROM player_play pp JOIN games g USING (game_id)
        WHERE g.season_type = 'REG' AND pp.play_nullified_by_penalty = 'N'
        """
    ).df()
    a = pd.crosstab(p["nfl_id"], p["align"], normalize="index").add_prefix("align_")
    a["snaps"] = p.groupby("nfl_id").size()
    r = p.dropna(subset=["route"])
    routes = pd.crosstab(r["nfl_id"], r["route"], normalize="index").add_prefix("route_")
    routes["route_CUT"] = routes[[f"route_{x}" for x in CUT_ROUTES]].sum(axis=1)
    routes["routes"] = r.groupby("nfl_id").size()
    return a.join(routes).reset_index()


def frame(use: Use, sh: pd.DataFrame) -> pd.DataFrame:
    drills = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    drills = drills[drills["drill"].isin(use.drills) & (drills["combine_position"] == use.position)]
    c = drills.groupby("nfl_id")[use.feature].mean().rename("c")
    meta = connect().execute(f"SELECT nfl_id, draft_year, {', '.join(matched.SIZE_SPEED)} FROM combine_results").df()
    count = "routes" if use.share.startswith("route_") else "snaps"
    d = meta.merge(c, on="nfl_id").merge(sh, on="nfl_id")
    return d[d[count] >= MIN_PLAYS].dropna(subset=["c", use.share])


def test(d: pd.DataFrame, share: str) -> dict:
    controls = matched.SIZE_SPEED + (["align_SLOT_WR"] if share.startswith("route_") else [])
    Z = d[controls].fillna(d[controls].median()).to_numpy(float)
    c, g = d["c"].to_numpy(), d[share].to_numpy()
    rho, p = stats.spearmanr(c, g)
    prho, pp = stats.spearmanr(evaluate._residualize(c, Z), evaluate._residualize(g, Z))
    by_class = {int(k): round(stats.spearmanr(x["c"], x[share])[0], 2) for k, x in d.groupby("draft_year") if len(x) >= 8}
    return {"n": len(d), "mean_share": g.mean(), "rho": rho, "p": p, "partial_rho": prho, "partial_p": pp,
            "by_class": by_class}


def table() -> pd.DataFrame:
    sh = shares()
    t = pd.DataFrame([{"position": u.position, "trait": u.trait, "share": u.share} | test(frame(u, sh), u.share)
                      for u in USES])
    t["partial_q"] = stats.false_discovery_control(t["partial_p"], method="bh")
    return t
