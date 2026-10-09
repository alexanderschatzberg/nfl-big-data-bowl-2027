"""Pooled test across all 510 players: does a Combine trait predict earning a role?

The outcome is playing time (starts and snaps per available season), which
exists for every player and position. The baseline is everything a team already
has on draft night: draft slot, stopwatch numbers, size and college production.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer

from . import conventional, evaluate
from .data import artifacts_dir, connect
from .screen import icc

BASE = ["forty", "ten_yd_split", "vertical", "broad_jump", "combine_height", "combine_weight",
        "ngs_college_production_score"]
OUTCOMES = ["start_rate", "snaps_per_season"]
GAMES_PER_SEASON = 17


def design() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.concat([conventional.frame(p).assign(position=p) for p in conventional.POSITIONS], ignore_index=True)
    starts = connect().execute("SELECT nfl_id, career_games_started FROM player_career_successes").df()
    df = df.merge(starts, on="nfl_id")
    seasons = conventional.LAST_SEASON + 1 - df["draft_year"]
    df["start_rate"] = df["career_games_started"] / (GAMES_PER_SEASON * seasons)
    df["start_rate"] -= df.groupby("position")["start_rate"].transform("mean")
    # Standardise within position so a pooled slope is a within-position slope.
    z = df.groupby("position")[BASE].transform(lambda v: (v - v.mean()) / v.std())
    dummies = pd.get_dummies(df[["position", "draft_year"]].astype(str), drop_first=True, dtype=float)
    base = np.column_stack([z.to_numpy(float), df[["log_pick"]].to_numpy(float), dummies.to_numpy()])
    return df, base


def scale(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean class-relative z across drills for each feature, with its cross-drill reliability."""
    pdz = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    scores = pdz.groupby("nfl_id")[features].mean().reset_index()
    rel = pd.DataFrame([dict(zip(("icc_single", "icc_mean", "n"), icc(pdz[f], pdz["nfl_id"])), feature=f)
                        for f in features]).sort_values("icc_mean", ascending=False)
    return scores, rel


def test(df: pd.DataFrame, base: np.ndarray, x: np.ndarray, outcome: str) -> dict:
    """Partial association of one trait with one outcome, overall and by slice."""
    y = df[outcome].to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    Z = SimpleImputer(strategy="median").fit_transform(base)
    rx, ry = evaluate._residualize(x[ok], Z[ok]), evaluate._residualize(y[ok], Z[ok])
    rho, p = stats.spearmanr(rx, ry)
    d = df[ok].assign(rx=rx, ry=ry)
    by = lambda col: {str(k): round(stats.spearmanr(g["rx"], g["ry"])[0], 2) for k, g in d.groupby(col)}  # noqa: E731
    cells = [stats.spearmanr(g["rx"], g["ry"])[0] for _, g in d.groupby(["position", "draft_year"]) if len(g) >= 10]
    w = np.ones(ok.sum())
    r2_base = evaluate.cv_r2(base[ok], y[ok], w, 10)
    return {"outcome": outcome, "n": int(ok.sum()), "partial_rho": rho, "p": p,
            "by_position": by("position"), "by_class": by("draft_year"),
            "cells_same_sign": f"{sum(np.sign(c) == np.sign(rho) for c in cells)}/{len(cells)}",
            "r2_base": r2_base,
            "delta_r2": evaluate.cv_r2(np.column_stack([base[ok], x[ok]]), y[ok], w, 10) - r2_base}
