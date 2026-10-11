"""Combine tracking -> per-attempt features -> class-relative player scores."""
import numpy as np
import pandas as pd

from .data import artifacts_dir, connect
from .kinematics import FEATURES, attempt_features

# The same drill is labelled differently across Combine years.
DRILL_ALIASES = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "LINE": "LINE_DRILL",
}
ATTEMPT_KEY = ["nfl_id", "event_id", "drill", "attempt"]
MIN_GROUP = 8  # players needed to z-score a drill within one draft class
Z_CLIP = 3.0
MIN_DURATION = 0.65  # of the drill-by-class median; a shorter clip was cut off before the drill finished


def build_attempt_features() -> pd.DataFrame:
    con = connect()
    df = con.execute(
        """
        SELECT nfl_id, event_id, drill_name, attempt, epoch(time) AS t, x, y, s
        FROM combine_tracking
        WHERE entity_type = 'PLAYER'
        ORDER BY nfl_id, event_id, drill_name, attempt, time
        """
    ).df()
    df["drill"] = df["drill_name"].replace(DRILL_ALIASES)
    rows = []
    for key, g in df.groupby(ATTEMPT_KEY, sort=False):
        feats = attempt_features(g["t"].values, g["x"].values, g["y"].values, g["s"].values,
                                 sprint=key[2] == "FORTY_YARD_DASH")
        if feats is not None:
            rows.append(dict(zip(ATTEMPT_KEY, key)) | feats)
    return pd.DataFrame(rows)


def _zscore(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[FEATURES]
    z = (df[FEATURES] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.where(g.transform("count") >= MIN_GROUP)


def build_player_drill(attempts: pd.DataFrame, min_duration: float = MIN_DURATION) -> pd.DataFrame:
    """One row per player x drill: median over attempts, z-scored against peers.

    Drill setups change between years (same name, different distance), so scores
    are relative to the same position group in the same draft class, falling
    back to all classes when a class is too small. Clips cut off early are
    dropped first: a stop drill that ends at top speed would score as no braking.
    """
    con = connect()
    meta = con.execute("SELECT nfl_id, draft_year, combine_position FROM combine_results").df()
    attempts = attempts.merge(meta[["nfl_id", "draft_year"]], on="nfl_id")
    typical = attempts.groupby(["drill", "draft_year"])["duration"].transform("median")
    attempts = attempts[attempts["duration"] >= min_duration * typical]
    pd_ = attempts.groupby(["nfl_id", "drill"], as_index=False)[FEATURES].median()
    pd_ = pd_.merge(meta, on="nfl_id")
    z = _zscore(pd_, ["drill", "combine_position", "draft_year"])
    z = z.fillna(_zscore(pd_, ["drill", "combine_position"])).clip(-Z_CLIP, Z_CLIP)
    return pd.concat([pd_[["nfl_id", "drill", "draft_year", "combine_position"]], z], axis=1)


def family_features(player_drill: pd.DataFrame, families: dict) -> pd.DataFrame:
    """Average z-scores across the drills of each family -> `family__feature` columns."""
    out = []
    for fam, (drills, feats) in families.items():
        sub = player_drill[player_drill["drill"].isin(drills)]
        agg = sub.groupby("nfl_id")[feats].mean()
        agg.columns = [f"{fam}__{c}" for c in feats]
        out.append(agg)
    return pd.concat(out, axis=1).reset_index()


def build() -> None:
    attempts = build_attempt_features()
    attempts.to_parquet(artifacts_dir() / "combine_attempts.parquet", index=False)
    build_player_drill(attempts).to_parquet(artifacts_dir() / "combine_player_drill.parquet", index=False)
