"""Wide interaction scan: every pair of predictors, against every outcome, per position.

`conventional.py` tests eleven pre-specified pairs. This scans all pairs drawn
from the stopwatch numbers, the tracked 40, on-field role and the drill
movement traits. The statistic is the t of the product term with both main
effects in the model. Thousands of tests on ~70 players will always turn up
small p-values, so the only numbers to read are `p_family` (min-p permutation
over the position's whole scan) and `p_tier` (the same within one tier), plus
whether the sign holds in each draft class.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from . import conventional, matched
from .combine import family_features
from .data import artifacts_dir

STOPWATCH = [c for c in conventional.PREDICTORS if c not in ("three_cone", "short_shuttle")]  # too few ran them
TRAITS = {
    "WR": {"stop": (matched._STOP, ["dec_win05", "brk_reacc_10", "brk_time_under"]),
           "cut": (matched._CUT, ["lat_acc", "curve_speed", "cut_turn_rate"]),
           "go": (["GO_ROUTE_RIGHT"], ["vmax", "accel_2to5"])},
    "TE": {"route": (["HOOK_ROUTE_RIGHT", "IN_ROUTE_LEFT"], ["dec_win05", "lat_acc"])},
    "OL": {"mirror": (matched._MIRROR, ["dec_win05", "lat_acc", "cod_rate"]),
           "set": (["PASS_RUSH_DROP"], ["acc_win05", "vmax"]),
           "pull": (matched._PULL, ["vmax", "accel_2to5"])},
    "DL": {"rush": (["PASS_RUSH_DRILL"], ["acc_win05", "vmax"]),
           "bend": (["RUN_THE_HOOP_DRILL"], ["lat_acc", "curve_speed"]),
           "agility": (matched._AGILITY, ["dec_win05", "cod_reacc"])},
    "DB": {"break": (matched._DB_BREAK, ["dec_win05", "acc_win10", "cut_turn_rate"])},
}
TRANSFER = {"WR": ["game_braking_oe"], "TE": ["game_braking_oe"], "OL": ["game_set_burst_oe"],
            "DL": ["game_burst_oe"], "DB": []}
TIERS = ["stopwatch x stopwatch", "stopwatch x drill", "drill x drill"]
MIN_N = 40
CLIP = 2.5  # products are leverage-prone; one extreme player should not make an interaction


def design(position: str) -> dict:
    df = conventional.frame(position)
    traits = family_features(pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet"), TRAITS[position])
    df = df.merge(traits, on="nfl_id", how="left")
    outs = conventional.OUTCOMES[position] + conventional.COMMON_OUTCOMES
    if TRANSFER[position]:
        extra = pd.read_parquet(artifacts_dir() / f"outcomes_{position}.parquet")[["nfl_id"] + TRANSFER[position]]
        df = df.merge(extra, on="nfl_id", how="left")
        df[TRANSFER[position]] -= df.groupby("role")[TRANSFER[position]].transform("mean")
        outs = outs + TRANSFER[position]
    played = df["role"] != "NONE"
    top_role = df.loc[played, "role"].mode()[0]
    df[f"is_{top_role}"] = np.where(played, (df["role"] == top_role).astype(float), np.nan)
    drill = [c for c in traits.columns if c != "nfl_id"]
    preds = STOPWATCH + [f"is_{top_role}"] + drill
    P = df[preds].to_numpy(float)
    P = np.clip((P - np.nanmean(P, axis=0)) / np.nanstd(P, axis=0), -CLIP, CLIP)
    pairs = []
    for i, j in combinations(range(len(preds)), 2):
        ok = ~np.isnan(P[:, i]) & ~np.isnan(P[:, j])
        X = np.column_stack([np.ones(len(P)), P[:, i], P[:, j], P[:, i] * P[:, j]])
        tier = TIERS[(preds[i] in drill) + (preds[j] in drill)]
        pairs.append((preds[i], preds[j], TIERS.index(tier), ok, X))
    return {"df": df, "pairs": pairs, "outcomes": outs, "Y": df[outs].to_numpy(float)}


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(coefficient, p) of the product term."""
    G = X.T @ X
    if np.linalg.cond(G) > 1e10:
        return np.nan, np.nan
    Gi = np.linalg.inv(G)
    beta = Gi @ (X.T @ y)
    dof = len(y) - 4
    se = np.sqrt(np.sum((y - X @ beta) ** 2) / dof * Gi[3, 3])
    return beta[3], 2 * stats.t.sf(abs(beta[3] / se), dof)


def _each(d: dict, Y: np.ndarray):
    for a, b, tier, ok, X in d["pairs"]:
        for k, name in enumerate(d["outcomes"]):
            m = ok & ~np.isnan(Y[:, k])
            if m.sum() >= MIN_N and np.ptp(Y[m, k]) > 0:
                yield a, b, tier, name, m, X[m], Y[m, k]


def scan(position: str) -> pd.DataFrame:
    d = design(position)
    year = d["df"]["draft_year"].to_numpy()
    rows = []
    for a, b, tier, name, m, X, y in _each(d, d["Y"]):
        y = (y - y.mean()) / y.std()
        beta, p = _fit(X, y)
        if not np.isfinite(p):
            continue
        by_class = {int(c): round(float(_fit(X[year[m] == c], y[year[m] == c])[0]), 2)
                    for c in np.unique(year[m]) if (year[m] == c).sum() >= 12}
        # Slope of `a` when `b` is one SD below and above its mean.
        lo, hi = np.linalg.lstsq(X, y, rcond=None)[0][[1, 3]] @ np.array([[1, 1], [-1, 1]])
        rows.append({"position": position, "tier": TIERS[tier], "a": a, "b": b, "outcome": name, "n": int(m.sum()),
                     "beta": beta, "p": p, "slope_a_low_b": lo, "slope_a_high_b": hi, "by_class": by_class})
    return pd.DataFrame(rows)


def min_p_null(position: str, seeds: list[int]) -> np.ndarray:
    """Smallest p per tier (columns) in each permuted scan (rows)."""
    d = design(position)
    out = np.ones((len(seeds), len(TIERS)))
    for r, s in enumerate(seeds):
        Y = d["Y"][np.random.default_rng(s).permutation(len(d["Y"]))]
        for _, _, tier, _, _, X, y in _each(d, Y):
            p = _fit(X, y)[1]
            if p < out[r, tier]:
                out[r, tier] = p
    return out


def adjust(table: pd.DataFrame, null: np.ndarray) -> pd.DataFrame:
    def fw(p, n):
        return (1 + np.searchsorted(np.sort(n), p, side="right")) / (1 + len(n))

    table = table.copy()
    table["p_family"] = fw(table["p"].to_numpy(), null.min(axis=1))
    table["p_tier"] = np.nan
    for k, tier in enumerate(TIERS):
        sel = table["tier"] == tier
        table.loc[sel, "p_tier"] = fw(table.loc[sel, "p"].to_numpy(), null[:, k])
    table["q"] = stats.false_discovery_control(table["p"], method="bh")
    return table.sort_values("p")
