"""Like-for-like transfer: does a Combine movement trait reappear in games?

Combine side: the player's mean class-relative z-score for a trait across every
drill they ran. Game side: the same trait computed on their regular-season
plays, summarised as a capability (90th percentile) and standardised within
on-field role. All positions are pooled, which is what gives this test power.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer

from . import evaluate
from .data import artifacts_dir, connect
from .hypotheses import STOPWATCH
from .screen import icc

TRAITS = ["vmax", "acc_max", "dec_max", "accel_2to5", "lat_acc", "curve_speed", "cut_turn_rate",
          "cut_radius", "cut_speed_loss", "cut_exit_gain", "brk_dec", "brk_reacc_10", "cod_reacc"]
LOWER_IS_MORE = {"cut_radius"}  # capability is the low end of the distribution
MIN_DRILLS = 3
MIN_PLAYS = 50
MIN_ROLE = 8


def combine_traits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global Combine traits and their consistency across different drills."""
    pdz = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    g = pdz.groupby("nfl_id")[TRAITS]
    traits = g.mean().where(g.count() >= MIN_DRILLS).add_prefix("c_").reset_index()
    rel = pd.DataFrame([dict(zip(("icc_single", "icc_mean", "n_icc"), icc(pdz[t], pdz["nfl_id"])), trait=t)
                        for t in TRAITS])
    return traits, rel


def game_traits() -> pd.DataFrame:
    con = connect()
    plays = pd.read_parquet(artifacts_dir() / "game_play_features.parquet", columns=["game_id", "play_id", "nfl_id"] + TRAITS)
    roles = con.execute(
        """
        SELECT nfl_id, mode(lined_up_position) AS role, any_value(combine_position) AS combine_position
        FROM player_play JOIN combine_results USING (nfl_id) GROUP BY nfl_id
        """
    ).df()
    g = plays.groupby("nfl_id")[TRAITS]
    cap = g.quantile(0.9)
    for t in LOWER_IS_MORE:
        cap[t] = plays.groupby("nfl_id")[t].quantile(0.1)
    cap = cap.where(g.count() >= MIN_PLAYS).reset_index().merge(roles, on="nfl_id")
    small = cap.groupby("role")["nfl_id"].transform("size") < MIN_ROLE
    cap.loc[small, "role"] = cap.loc[small, "combine_position"]
    z = cap.groupby("role")[TRAITS].transform(lambda v: (v - v.mean()) / v.std())
    return pd.concat([cap[["nfl_id", "role"]], z.add_prefix("g_")], axis=1)


def design() -> tuple[pd.DataFrame, list[str]]:
    meta = connect().execute(
        """
        SELECT p.nfl_id, p.display_name, p.draft_year, p.draft_overall_pick, cr.* EXCLUDE (nfl_id, draft_year)
        FROM players p JOIN combine_results cr USING (nfl_id)
        """
    ).df()
    traits, _ = combine_traits()
    df = meta.merge(traits, on="nfl_id").merge(game_traits(), on="nfl_id")
    dummies = pd.get_dummies(df[["draft_year", "role"]].astype(str), drop_first=True, dtype=float)
    return pd.concat([df, dummies], axis=1), STOPWATCH + list(dummies.columns)


def table(repeats: int = 5) -> pd.DataFrame:
    df, base_cols = design()
    _, rel = combine_traits()
    Z = SimpleImputer(strategy="median").fit_transform(df[base_cols])
    rows = []
    for t in TRAITS:
        ok = (df[f"c_{t}"].notna() & df[f"g_{t}"].notna()).to_numpy()
        x, y, z = df.loc[ok, f"c_{t}"].to_numpy(), df.loc[ok, f"g_{t}"].to_numpy(), Z[ok]
        rho, p = stats.spearmanr(x, y)
        prho, pp = stats.spearmanr(evaluate._residualize(x, z), evaluate._residualize(y, z))
        base = df.loc[ok, base_cols].to_numpy(float)
        w = np.ones(ok.sum())
        r2_base = evaluate.cv_r2(base, y, w, repeats)
        r2_full = evaluate.cv_r2(np.column_stack([base, x]), y, w, repeats)
        rows.append({"trait": t, "n": int(ok.sum()), "rho": rho, "p": p, "partial_rho": prho, "partial_p": pp,
                     "r2_stopwatch": r2_base, "delta_r2": r2_full - r2_base})
    out = pd.DataFrame(rows).merge(rel[["trait", "icc_mean"]], on="trait")
    out["partial_q"] = stats.false_discovery_control(out["partial_p"], method="bh")
    return out.sort_values("partial_p")
