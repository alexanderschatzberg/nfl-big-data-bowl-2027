"""Matched transfer: the same trait, in the drill and in the game situation it rehearses.

Each pair names the Combine drills, the game plays they correspond to, and one
movement feature measured identically on both sides. The game side is a
capability (an upper quantile over the player's plays), and the partial
correlation removes 40 time, size and on-field role, so a hit means the drill
tracking says something the stopwatch does not.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate
from .data import artifacts_dir, connect

SIZE_SPEED = ["forty", "combine_weight", "combine_height"]
BREAKING = ("HITCH", "OUT", "IN", "SLANT", "CORNER", "POST")
_STOP = ["COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT"]
_CUT = ["SPEED_OUT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "SLANT_ROUTE_LEFT", "POST_CORNER_ROUTE_LEFT",
        "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT"]
_AGILITY = ["FOUR_BAG_AGILITY_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "BODY_CONTROL_DRILL"]
_MIRROR = ["PASS_PRO_MIRROR_DRILL", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"]
_PULL = ["OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT", "OL_PULL_DRILL_LONG_PULL_RIGHT",
         "SCREEN_DRILL"]
_DB_BREAK = ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "W_DRILL",
             "BOX_DRILL"]


@dataclass(frozen=True)
class Pair:
    position: str
    name: str
    drills: list
    feature: str
    plays: str  # pandas query on the game-play table
    quantile: float = 0.9
    min_plays: int = 20


PAIRS = [
    Pair("WR", "go-route drill top speed", ["GO_ROUTE_RIGHT"], "vmax", "route_ran == 'GO'"),
    Pair("WR", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "route_ran == 'GO'"),
    Pair("WR", "stop-route braking", _STOP, "dec_win05", "route_ran == 'HITCH'"),
    Pair("WR", "stop-route re-acceleration", _STOP, "brk_reacc_10", "route_ran == 'HITCH'", 0.5),
    Pair("WR", "stop-route time in break", _STOP, "brk_time_under", "route_ran == 'HITCH'", 0.5),
    Pair("WR", "cut lateral acceleration", _CUT, "lat_acc", "route_ran in ['OUT','IN','SLANT','CORNER','POST']"),
    Pair("WR", "cut turn rate", _CUT, "cut_turn_rate", "route_ran in ['OUT','IN','SLANT','CORNER','POST']"),
    Pair("WR", "cut speed carried", _CUT, "curve_speed", "route_ran in ['OUT','IN','SLANT','CORNER','POST']", 0.5),
    Pair("DL", "pass-rush drill burst", ["PASS_RUSH_DRILL"], "acc_win05", "rush", min_plays=30),
    Pair("DL", "pass-rush drill top speed", ["PASS_RUSH_DRILL"], "vmax", "rush", min_plays=30),
    Pair("DL", "hoop lateral acceleration", ["RUN_THE_HOOP_DRILL"], "lat_acc", "rush", min_plays=30),
    Pair("DL", "hoop curve speed", ["RUN_THE_HOOP_DRILL"], "curve_speed", "rush", 0.5, 30),
    Pair("DL", "agility braking", _AGILITY, "dec_win05", "rush", min_plays=30),
    Pair("OL", "pass-set burst", ["PASS_RUSH_DROP"], "acc_win05", "pass_block", min_plays=30),
    Pair("OL", "pass-set top speed", ["PASS_RUSH_DROP"], "vmax", "pass_block", min_plays=30),
    Pair("OL", "mirror braking", _MIRROR, "dec_win05", "pass_block", min_plays=30),
    Pair("OL", "mirror lateral acceleration", _MIRROR, "lat_acc", "pass_block", min_plays=30),
    Pair("OL", "pull top speed", _PULL, "vmax", "not pass_block", min_plays=30),
    Pair("DB", "break-drill braking", _DB_BREAK, "dec_win05", "cover", min_plays=30),
    Pair("DB", "break-drill burst", _DB_BREAK, "acc_win10", "cover", min_plays=30),
    Pair("DB", "break-drill turn rate", _DB_BREAK, "cut_turn_rate", "cover", min_plays=30),
    Pair("DB", "40 tracked top speed", ["FORTY_YARD_DASH"], "vmax", "cover", 0.99, 30),
    # Sharper versions of pairs whose game side was too loose the first time.
    Pair("OL", "pull top speed, lateral run plays", _PULL, "vmax", "not pass_block and lateral >= 4", min_plays=10),
    Pair("OL", "pull burst, lateral run plays", _PULL, "acc_win10", "not pass_block and lateral >= 4", min_plays=10),
    Pair("WR", "40 start burst vs press", ["FORTY_YARD_DASH"], "acc_win10", "cushion <= 3", min_plays=15),
    Pair("WR", "go-drill start burst vs press", ["GO_ROUTE_RIGHT"], "acc_win10", "cushion <= 3", min_plays=15),
    Pair("DL", "hoop lateral acceleration, edge rushes", ["RUN_THE_HOOP_DRILL"], "lat_acc",
         "rush and lined_up == 'EDGE'", min_plays=30),
    Pair("DL", "hoop curve speed, edge rushes", ["RUN_THE_HOOP_DRILL"], "curve_speed",
         "rush and lined_up == 'EDGE'", 0.5, 30),
    Pair("DL", "pass-rush drill burst, edge rushes", ["PASS_RUSH_DRILL"], "acc_win05",
         "rush and lined_up == 'EDGE'", min_plays=30),
]
LATERAL_WINDOW = 1.6  # seconds after the snap in which a pulling or stretching lineman has moved sideways


def game_plays() -> pd.DataFrame:
    con = connect()
    plays = con.execute(
        """
        SELECT game_id, play_id, nfl_id, combine_position AS position, lined_up_position AS lined_up, route_ran,
               cushion, g.season, player_get_off IS NOT NULL AS rush, pressure_allowed IS NOT NULL AS pass_block,
               coverage_assignment IS NOT NULL AS cover
        FROM player_play JOIN combine_results USING (nfl_id) JOIN games g USING (game_id)
        """
    ).df()
    # Sideways travel just after the snap. Game tracking has no teammates to tell a pull from
    # a stretch block, so linemen's "lateral run plays" are both.
    lateral = con.execute(
        f"""
        WITH snap AS (SELECT game_id, play_id, nfl_id, min(time) AS ts, arg_min(y, time) AS y0
                      FROM game_tracking WHERE event = 'ball_snap' GROUP BY ALL)
        SELECT t.game_id, t.play_id, t.nfl_id, max(abs(t.y - s.y0)) AS lateral
        FROM game_tracking t JOIN snap s USING (game_id, play_id, nfl_id)
        WHERE t.time BETWEEN s.ts AND s.ts + INTERVAL '{LATERAL_WINDOW} seconds'
        GROUP BY ALL
        """
    ).df()
    key = ["game_id", "play_id", "nfl_id"]
    return (pd.read_parquet(artifacts_dir() / "game_play_features.parquet").merge(plays, on=key)
            .merge(lateral, on=key, how="left"))


def pair_frame(pair: Pair, plays: pd.DataFrame, **overrides) -> pd.DataFrame:
    """One row per player: Combine score `c`, game capability `g`, and controls."""
    q = overrides.get("quantile", pair.quantile)
    min_plays = overrides.get("min_plays", pair.min_plays)
    feature = overrides.get("feature", pair.feature)
    drills = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    drills = drills[drills["drill"].isin(overrides.get("drills", pair.drills)) & (drills["combine_position"] == pair.position)]
    c = drills.groupby("nfl_id")[feature].mean().rename("c")
    sub = plays[plays["position"] == pair.position].query(overrides.get("where", pair.plays))
    g = sub.groupby("nfl_id")[feature]
    g = g.quantile(q).where(g.count() >= min_plays).rename("g")
    role = plays.dropna(subset=["lined_up"]).groupby("nfl_id")["lined_up"].agg(lambda s: s.mode().iat[0]).rename("role")
    meta = connect().execute(f"SELECT nfl_id, draft_year, {', '.join(SIZE_SPEED)} FROM combine_results").df()
    return meta.merge(c, on="nfl_id").merge(g, on="nfl_id").merge(role, on="nfl_id").dropna(subset=["c", "g"])


def test(d: pd.DataFrame) -> dict:
    controls = d[SIZE_SPEED].fillna(d[SIZE_SPEED].median()).to_numpy(float)
    Z = np.column_stack([controls, pd.get_dummies(d["role"], drop_first=True, dtype=float).to_numpy()])
    rho, p = stats.spearmanr(d["c"], d["g"])
    prho, pp = stats.spearmanr(evaluate._residualize(d["c"].to_numpy(), Z), evaluate._residualize(d["g"].to_numpy(), Z))
    by_class = {int(k): round(stats.spearmanr(g["c"], g["g"])[0], 2) for k, g in d.groupby("draft_year") if len(g) >= 8}
    return {"n": len(d), "rho": rho, "p": p, "partial_rho": prho, "partial_p": pp, "by_class": by_class}


def table() -> pd.DataFrame:
    plays = game_plays()
    rows = [{"position": pr.position, "trait": pr.name} | test(pair_frame(pr, plays)) for pr in PAIRS]
    t = pd.DataFrame(rows)
    t["partial_q"] = stats.false_discovery_control(t["partial_p"], method="bh")
    return t


def _partial(d: pd.DataFrame) -> float:
    return test(d)["partial_rho"]


def robustness(name: str = "stop-route braking", n_boot: int = 2000) -> pd.DataFrame:
    """The headline pair under every reasonable variation, plus placebos that should be null."""
    plays = game_plays()
    pair = next(p for p in PAIRS if p.name == name)
    variants = {"as specified": {}}
    variants |= {f"game quantile {q}": {"quantile": q} for q in (0.5, 0.75, 0.85, 0.95)}
    variants |= {f"min plays {m}": {"min_plays": m} for m in (10, 30, 40)}
    variants |= {f"feature {f}": {"feature": f} for f in ("dec_win10", "dec_p95", "dec_max") if f != pair.feature}
    variants |= {f"drill {d} only": {"drills": [d]} for d in pair.drills}
    variants |= {"all breaking routes": {"where": f"route_ran in {list(BREAKING)}"},
                 "placebo: go routes in games": {"where": "route_ran == 'GO'"},
                 "placebo: go-route drill": {"drills": ["GO_ROUTE_RIGHT"]},
                 "placebo: 40-yard dash": {"drills": ["FORTY_YARD_DASH"]}}
    rows = [{"variant": v} | test(pair_frame(pair, plays, **o)) for v, o in variants.items()]
    d = pair_frame(pair, plays).reset_index(drop=True)
    rng = np.random.default_rng(0)
    boot = [_partial(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    null = [_partial(d.assign(c=rng.permutation(d["c"].to_numpy()))) for _ in range(n_boot)]
    t = pd.DataFrame(rows)
    t.attrs = {"ci95": np.percentile(boot, [2.5, 97.5]).round(3).tolist(),
               "perm_p": (1 + np.sum(np.abs(null) >= abs(_partial(d)))) / (1 + n_boot)}
    return t


def benchmark(name: str = "stop-route braking", season_min: int = 12) -> dict:
    """How good is the drill as a predictor, next to the best available yardstick?

    The yardstick is the same game measure from the player's own previous
    season. Also reports the reliability of each side and the transfer
    correlation corrected for it.
    """
    plays = game_plays()
    pair = next(p for p in PAIRS if p.name == name)
    d = pair_frame(pair, plays)
    sub = plays[(plays["position"] == pair.position)].query(pair.plays)
    sub = sub[sub["nfl_id"].isin(d["nfl_id"])]
    g = sub.groupby(["nfl_id", "season"])[pair.feature]
    by_season = g.quantile(pair.quantile).where(g.count() >= season_min).unstack()
    pairs = pd.concat([by_season[[a, b]].set_axis(["prev", "next"], axis=1).dropna()
                       for a, b in zip(by_season.columns[:-1], by_season.columns[1:])])
    # Split-half reliability of the game capability (random halves of each player's plays).
    rng = np.random.default_rng(0)
    halves = []
    for _ in range(200):
        h = rng.integers(0, 2, len(sub))
        a = sub[h == 0].groupby("nfl_id")[pair.feature].quantile(pair.quantile)
        b = sub[h == 1].groupby("nfl_id")[pair.feature].quantile(pair.quantile)
        halves.append(stats.spearmanr(*a.align(b, join="inner"))[0])
    r_half = float(np.mean(halves))
    game_rel = 2 * r_half / (1 + r_half)
    drills = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    drills = drills[drills["drill"].isin(pair.drills) & drills["nfl_id"].isin(d["nfl_id"])]
    wide = drills.pivot(index="nfl_id", columns="drill", values=pair.feature).dropna()
    r_drill = stats.spearmanr(wide.iloc[:, 0], wide.iloc[:, 1])[0] if wide.shape[1] == 2 else np.nan
    drill_rel = 2 * r_drill / (1 + r_drill)
    rho = stats.spearmanr(d["c"], d["g"])[0]
    return {"n": len(d), "combine_to_game": rho, "season_to_next_season": stats.spearmanr(pairs["prev"], pairs["next"])[0],
            "season_pairs": len(pairs), "game_reliability": game_rel, "drill_to_drill": r_drill,
            "drill_reliability": drill_rel, "corrected_for_reliability": rho / np.sqrt(game_rel * drill_rel)}
