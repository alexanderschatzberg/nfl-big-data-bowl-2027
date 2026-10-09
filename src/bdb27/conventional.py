"""Do the numbers scouts already trust predict NFL success? And do they interact?

Every standard Combine measurement, plus the tracking-derived versions of the
40, is tested against several success measures per position. Family-wise
error is controlled by a min-p permutation test over the whole family of
tests for a position, so "nothing here" and "this one is real" are both
statements about the family, not about a single cherry-picked pair.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .data import artifacts_dir, connect

POSITIONS = ["WR", "TE", "OL", "DL", "DB"]
CONVENTIONAL = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "bench_reps",
                "combine_height", "combine_weight", "arm_length", "hand_size",
                "ngs_athleticism_score", "ngs_college_production_score", "log_pick"]
TRACKING = ["trk_40_top_speed", "trk_40_initial_accel"]
PREDICTORS = CONVENTIONAL + TRACKING
OUTCOMES = {
    "WR": ["sep_oe_all", "separation_raw", "yards_per_route", "targets_per_route"],
    "TE": ["sep_oe_all", "separation_raw", "yards_per_route", "targets_per_route"],
    "OL": ["pressure_allowed_oe", "pressure_allowed_raw", "peak_pressure_prob_oe"],
    "DL": ["get_off_oe", "get_off_raw", "pressure_rate_oe", "pressure_rate_raw", "sack_rate_raw"],
    "DB": [],
}
COMMON_OUTCOMES = ["game_speed_p99", "snaps_per_season"]
INTERACTIONS = [("forty", "combine_weight"), ("ten_yd_split", "combine_weight"), ("vertical", "combine_weight"),
                ("broad_jump", "combine_weight"), ("forty", "combine_height"), ("forty", "ten_yd_split"),
                ("trk_40_top_speed", "combine_weight"), ("trk_40_initial_accel", "combine_weight"),
                ("trk_40_top_speed", "trk_40_initial_accel"), ("log_pick", "forty"),
                ("log_pick", "ngs_athleticism_score")]
MIN_N = 30
MIN_ROLE = 8
LAST_SEASON = 2025
UNDRAFTED_PICK = 260


def frame(position: str) -> pd.DataFrame:
    con = connect()
    df = con.execute(
        """
        SELECT p.nfl_id, p.display_name, p.draft_year, p.draft_overall_pick,
               cr.* EXCLUDE (nfl_id, draft_year),
               s.career_offensive_snaps + s.career_defensive_snaps AS snaps
        FROM players p JOIN combine_results cr USING (nfl_id) JOIN player_career_successes s USING (nfl_id)
        WHERE cr.combine_position = ?
        """,
        [position],
    ).df()
    df["log_pick"] = np.log(df["draft_overall_pick"].fillna(UNDRAFTED_PICK).astype(float))
    df["snaps_per_season"] = np.log1p(df["snaps"] / (LAST_SEASON + 1 - df["draft_year"]))
    att = pd.read_parquet(artifacts_dir() / "combine_attempts.parquet")
    forty = att[att["drill"] == "FORTY_YARD_DASH"].groupby("nfl_id")[["f40_vmax", "f40_a0"]].max()
    forty.columns = TRACKING
    plays = pd.read_parquet(artifacts_dir() / "game_play_features.parquet", columns=["nfl_id", "vmax"])
    g = plays.groupby("nfl_id")["vmax"]
    speed = g.quantile(0.99).where(g.count() >= 50).rename("game_speed_p99")
    df = df.merge(forty, on="nfl_id", how="left").merge(speed, on="nfl_id", how="left")
    path = artifacts_dir() / f"outcomes_{position}.parquet"
    if path.exists():
        df = df.merge(pd.read_parquet(path)[["nfl_id"] + OUTCOMES[position]], on="nfl_id", how="left")
    # Tackles and guards, or slot and outside receivers, face different jobs. Without
    # this, body-type measurements just proxy for role (long arms -> tackle -> more pressure).
    roles = con.execute("SELECT nfl_id, mode(lined_up_position) AS role FROM player_play GROUP BY nfl_id").df()
    df = df.merge(roles, on="nfl_id", how="left")
    df["role"] = df["role"].fillna("NONE")
    df.loc[df.groupby("role")["nfl_id"].transform("size") < MIN_ROLE, "role"] = "OTHER"
    outs = OUTCOMES[position] + COMMON_OUTCOMES
    df[outs] = df[outs] - df.groupby("role")[outs].transform("mean")
    return df


def _tests(position: str) -> list[tuple]:
    outs = OUTCOMES[position] + COMMON_OUTCOMES
    return ([("main", a, None, y) for a in PREDICTORS for y in outs]
            + [("interaction", a, b, y) for a, b in INTERACTIONS for y in outs])


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def _stat(kind: str, a: np.ndarray, b: np.ndarray | None, y: np.ndarray) -> tuple[float, float]:
    """(effect, p). Main effect: Spearman rho. Interaction: standardised product coefficient."""
    if kind == "main":
        return stats.spearmanr(a, y)
    X = np.column_stack([np.ones(len(y)), _z(a), _z(b), _z(a) * _z(b)])
    beta, res, *_ = np.linalg.lstsq(X, _z(y), rcond=None)
    dof = len(y) - 4
    sigma2 = np.sum((_z(y) - X @ beta) ** 2) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[3, 3])
    return beta[3], 2 * stats.t.sf(abs(beta[3] / se), dof)


def run(position: str, perm: np.ndarray | None = None) -> pd.DataFrame:
    """All tests for a position. `perm` reorders which player owns which outcomes."""
    df = frame(position)
    outs = OUTCOMES[position] + COMMON_OUTCOMES
    Y = df[outs].to_numpy(float)
    if perm is not None:
        Y = Y[perm]
    rows = []
    for kind, a, b, y in _tests(position):
        yv = Y[:, outs.index(y)]
        cols = [df[a].to_numpy(float)] + ([df[b].to_numpy(float)] if b else [])
        ok = ~np.isnan(yv) & np.all([~np.isnan(c) for c in cols], axis=0)
        if ok.sum() < MIN_N:
            continue
        if any(np.ptp(c[ok]) == 0 for c in cols):
            continue
        effect, p = _stat(kind, cols[0][ok], cols[1][ok] if b else None, yv[ok])
        if np.isfinite(p):
            rows.append({"position": position, "kind": kind, "a": a, "b": b, "outcome": y,
                         "n": int(ok.sum()), "effect": effect, "p": p})
    return pd.DataFrame(rows)


def min_p_null(position: str, seeds: list[int]) -> list[float]:
    n = len(frame(position))
    return [float(run(position, np.random.default_rng(s).permutation(n))["p"].min()) for s in seeds]


def adjust(table: pd.DataFrame, null_min_p: list[float]) -> pd.DataFrame:
    """Family-wise p: how often does the best of a permuted family beat this test?"""
    null = np.sort(np.asarray(null_min_p))
    table = table.copy()
    table["p_family"] = (1 + np.searchsorted(null, table["p"].to_numpy(), side="right")) / (1 + len(null))
    table["q"] = stats.false_discovery_control(table["p"], method="bh")
    return table.sort_values("p")
