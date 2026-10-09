"""Regular-season outcomes, context-adjusted at the play level, then shrunk per player.

Every outcome is oriented so that higher is better for the player.
"""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import artifacts_dir, connect

MIN_PLAYS = 50
BREAK_ROUTES = ("HITCH", "IN", "OUT", "CORNER", "POST", "SLANT", "CROSS")

CONTEXT_NUM = ["down", "yards_to_go", "quarter", "is_home"]
CONTEXT_CAT = ["lined_up_position", "offense_formation", "receiver_alignment",
               "team_coverage_man_zone", "team_coverage_type"]


def build_game_kinematics() -> None:
    """Per player-play movement after the snap, from game tracking."""
    con = connect()
    con.execute(
        f"""
        COPY (
            WITH snap AS (
                SELECT game_id, play_id, nfl_id, min(time) AS ts
                FROM game_tracking WHERE event = 'ball_snap' GROUP BY ALL
            ), f AS (
                SELECT g.game_id, g.play_id, g.nfl_id, x, y, s,
                       round(epoch(g.time - snap.ts), 1) AS dt,
                       (lag(s, 2) OVER w - lead(s, 2) OVER w) / 0.4 AS braking
                FROM game_tracking g JOIN snap USING (game_id, play_id, nfl_id)
                WHERE g.time >= snap.ts
                WINDOW w AS (PARTITION BY g.game_id, g.play_id, g.nfl_id ORDER BY g.time)
            )
            SELECT game_id, play_id, nfl_id,
                   max(s) AS game_vmax,
                   max(braking) AS game_dec_max,
                   max(-braking) AS game_acc_max,
                   sqrt((max(x) FILTER (dt = 1.0) - max(x) FILTER (dt = 0.0)) ^ 2
                      + (max(y) FILTER (dt = 1.0) - max(y) FILTER (dt = 0.0)) ^ 2) AS game_burst_1s
            FROM f GROUP BY ALL
        ) TO '{artifacts_dir() / "game_kinematics.parquet"}' (FORMAT parquet)
        """
    )


def load_plays(position: str) -> pd.DataFrame:
    con = connect()
    return con.execute(
        f"""
        SELECT pp.*, (pp.team_abbr = g.home_team_abbr)::INT AS is_home,
               k.game_vmax, k.game_dec_max, k.game_acc_max, k.game_burst_1s
        FROM player_play pp
        JOIN games g USING (game_id)
        JOIN combine_results cr USING (nfl_id)
        LEFT JOIN read_parquet('{artifacts_dir() / "game_kinematics.parquet"}') k
               USING (game_id, play_id, nfl_id)
        WHERE g.season_type = 'REG' AND cr.combine_position = ?
          AND pp.play_nullified_by_penalty = 'N'
        """,
        [position],
    ).df()


def over_expected(plays: pd.DataFrame, y: str, num: list[str], cat: list[str], binary: bool = False) -> pd.Series:
    """Actual minus expected, with the expectation cross-fitted by player.

    A player's own plays never train the model that sets their expectation.
    """
    num = [c for c in num if plays[c].notna().any()]  # some context is role-specific
    X = plays[num + cat].copy()
    X[num] = X[num].astype(float)
    X[cat] = X[cat].astype(object).fillna("MISSING").astype(str)
    yv = plays[y].astype(float).values
    cat_idx = list(range(len(num), len(num) + len(cat)))
    Model = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
    exp = np.empty(len(plays))
    n_splits = min(5, plays["nfl_id"].nunique())
    for tr, te in GroupKFold(n_splits).split(X, yv, plays["nfl_id"]):
        enc = make_column_transformer(
            ("passthrough", num),
            (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
        )
        model = make_pipeline(enc, Model(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=50, categorical_features=cat_idx))
        model.fit(X.iloc[tr], yv[tr])
        exp[te] = model.predict_proba(X.iloc[te])[:, 1] if binary else model.predict(X.iloc[te])
    return pd.Series(yv - exp, index=plays.index)


def player_aggregate(plays: pd.DataFrame, oe: pd.Series, name: str, sign: float = 1.0) -> pd.DataFrame:
    """Empirical-Bayes shrink each player's mean toward zero by its reliability."""
    d = pd.DataFrame({"nfl_id": plays["nfl_id"].values, "oe": sign * oe.values}).dropna()
    g = d.groupby("nfl_id")["oe"].agg(["mean", "var", "count"])
    g = g[g["count"] >= MIN_PLAYS]
    within = np.average(g["var"], weights=g["count"] - 1)
    between = max(g["mean"].var() - (within / g["count"]).mean(), 0.0)
    rel = between / (between + within / g["count"]) if between > 0 else g["count"] * 0.0
    return pd.DataFrame({name: g["mean"] * rel, f"{name}__n": g["count"], f"{name}__rel": rel}).reset_index()


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="nfl_id", how="outer")
    return out


def raw_rate(plays: pd.DataFrame, values: pd.Series, name: str, sign: float = 1.0) -> pd.DataFrame:
    """An unadjusted per-play rate, shrunk toward the position average."""
    v = values.astype(float)
    return player_aggregate(plays, v - v.mean(), name, sign)


def wr_outcomes(position: str = "WR") -> pd.DataFrame:
    routes = load_plays(position)
    routes = routes[routes["route_ran"].notna()].reset_index(drop=True)
    p = routes[routes["separation_at_pass_forward"].notna()].reset_index(drop=True)
    num = CONTEXT_NUM + ["cushion", "dropback_duration"]
    cat = CONTEXT_CAT + ["route_ran"]
    brk = p[p["route_ran"].isin(BREAK_ROUTES)].reset_index(drop=True)
    brk_k = brk[brk["game_dec_max"].notna()].reset_index(drop=True)
    return _combine([
        player_aggregate(brk, over_expected(brk, "separation_at_pass_forward", num, cat), "sep_oe_break"),
        player_aggregate(p, over_expected(p, "separation_at_pass_forward", num, cat), "sep_oe_all"),
        player_aggregate(brk_k, over_expected(brk_k, "game_dec_max", num, cat), "game_braking_oe"),
        raw_rate(routes, routes["rec_yards"].fillna(0), "yards_per_route"),
        raw_rate(routes, routes["target"], "targets_per_route"),
        raw_rate(p, p["separation_at_pass_forward"], "separation_raw"),
    ])


def ol_outcomes() -> pd.DataFrame:
    p = load_plays("OL")
    p = p[p["pressure_allowed"].notna()].reset_index(drop=True)
    num = CONTEXT_NUM + ["pass_rushers_encountered", "dropback_duration"]
    pk = p[p["peak_pressure_probability_allowed"].notna()].reset_index(drop=True)
    p["pressure_allowed"] = p["pressure_allowed"].astype(float)
    k = p[p["game_burst_1s"].notna()].reset_index(drop=True)
    return _combine([
        player_aggregate(k, over_expected(k, "game_burst_1s", num, CONTEXT_CAT), "game_set_burst_oe"),
        player_aggregate(p, over_expected(p, "pressure_allowed", num, CONTEXT_CAT, binary=True),
                         "pressure_allowed_oe", sign=-1),
        player_aggregate(pk, over_expected(pk, "peak_pressure_probability_allowed", num, CONTEXT_CAT),
                         "peak_pressure_prob_oe", sign=-1),
        raw_rate(p, p["pressure_allowed"], "pressure_allowed_raw", sign=-1),
    ])


def dl_outcomes() -> pd.DataFrame:
    p = load_plays("DL")
    p = p[p["player_get_off"].notna()].reset_index(drop=True)
    p["blitzing"] = p["blitzing"].astype(float)
    p["pressure"] = p["time_to_pressure"].notna().astype(float)
    num = CONTEXT_NUM + ["blitzing"]
    pk = p[p["game_burst_1s"].notna()].reset_index(drop=True)
    return _combine([
        player_aggregate(p, over_expected(p, "player_get_off", num, CONTEXT_CAT), "get_off_oe", sign=-1),
        player_aggregate(p, over_expected(p, "pressure", num + ["dropback_duration"], CONTEXT_CAT, binary=True),
                         "pressure_rate_oe"),
        player_aggregate(pk, over_expected(pk, "game_burst_1s", num, CONTEXT_CAT), "game_burst_oe"),
        raw_rate(p, p["pressure"], "pressure_rate_raw"),
        raw_rate(p, p["sack"].fillna(0), "sack_rate_raw"),
        raw_rate(p, p["player_get_off"], "get_off_raw", sign=-1),
    ])


BUILDERS = {"WR": wr_outcomes, "TE": lambda: wr_outcomes("TE"), "OL": ol_outcomes, "DL": dl_outcomes}


def build(kinematics: bool = True) -> None:
    if kinematics:
        build_game_kinematics()
    for pos, fn in BUILDERS.items():
        fn().to_parquet(artifacts_dir() / f"outcomes_{pos}.parquet", index=False)
