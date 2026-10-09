"""Cheap screening experiments that decide what the expensive runs test.

For every candidate trait: is it repeatable (stage 1), is it more than the
stopwatch (stage 1), does it reappear in games (stage 2), and does it relate
to results (stage 3)? None of this needs permutations or model search.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate
from .combine import _zscore
from .data import artifacts_dir, connect
from .hypotheses import Hypothesis, load_design

MIN_ICC = 0.4
MAX_STOPWATCH_R2 = 0.5


def icc(values: pd.Series, groups: pd.Series) -> tuple[float, float, int]:
    """One-way random-effects ICC: (single measurement, player mean, players used)."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    counts = d.groupby("g")["v"].transform("count")
    d = d[counts >= 2]
    n, N = d["g"].nunique(), len(d)
    if n < 10:
        return np.nan, np.nan, n
    g = d.groupby("g")["v"]
    k = g.count()
    msb = (k * (g.mean() - d["v"].mean()) ** 2).sum() / (n - 1)
    msw = ((d["v"] - g.transform("mean")) ** 2).sum() / (N - n)
    k0 = (N - (k**2).sum() / N) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), (msb - msw) / msb, n


def attempt_zscores() -> pd.DataFrame:
    attempts = pd.read_parquet(artifacts_dir() / "combine_attempts.parquet")
    meta = connect().execute("SELECT nfl_id, draft_year, combine_position FROM combine_results").df()
    attempts = attempts.merge(meta, on="nfl_id")
    z = _zscore(attempts, ["drill", "combine_position", "draft_year"])
    return pd.concat([attempts[["nfl_id", "drill"]], z], axis=1)


def screen(hyp: Hypothesis) -> pd.DataFrame:
    design = load_design(hyp)
    df = design.df
    az = attempt_zscores()
    az = az[az["nfl_id"].isin(df["nfl_id"])]
    stopwatch = df[design.stopwatch].to_numpy(float)
    rows = []
    for fam, (drills, feats) in hyp.families.items():
        sub = az[az["drill"].isin(drills)]
        for feat in feats:
            col = f"{fam}__{feat}"
            x = df[col].to_numpy(float)
            ok = ~np.isnan(x)
            icc1, icck, n_icc = icc(sub[feat], sub["nfl_id"])
            row = {"trait": col, "n": int(ok.sum()), "icc_single": icc1, "icc_mean": icck, "n_icc": n_icc,
                   "r2_from_stopwatch": evaluate.cv_r2(stopwatch[ok], x[ok], np.ones(ok.sum()), repeats=5)
                   if ok.sum() >= 30 else np.nan}
            for outcome in hyp.outcomes:
                both = ok & df[outcome].notna().to_numpy()
                rho, p = stats.spearmanr(x[both], df.loc[both, outcome]) if both.sum() >= 20 else (np.nan, np.nan)
                row[f"rho__{outcome}"], row[f"p__{outcome}"] = rho, p
            rows.append(row)
    t = pd.DataFrame(rows)
    # Unknown repeatability (one attempt of one drill) is not a failure, but it is flagged.
    t["repeatable"] = np.where(t["icc_mean"].isna(), "unknown", np.where(t["icc_mean"] >= MIN_ICC, "yes", "no"))
    t["beyond_stopwatch"] = t["r2_from_stopwatch"] <= MAX_STOPWATCH_R2
    t["passes_stage1"] = (t["repeatable"] != "no") & t["beyond_stopwatch"]
    return t
