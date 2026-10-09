"""The same per-clip movement features as the Combine, computed on game plays.

Like-for-like measurement is what makes a transfer test meaningful: a trait is
compared with itself, not with a loosely related game statistic.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import artifacts_dir, connect
from .kinematics import attempt_features

KEY = ["game_id", "play_id", "nfl_id"]


def _features(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, g in df.groupby(KEY, sort=False):
        f = attempt_features(g["t"].values, g["x"].values, g["y"].values, g["s"].values)
        if f is not None:
            rows.append(dict(zip(KEY, key)) | f)
    return rows


def build(n_jobs: int = -1) -> None:
    con = connect()
    out = []
    for season in (2023, 2024, 2025):  # one season at a time to bound memory
        df = con.execute(
            f"""
            WITH snap AS (
                SELECT game_id, play_id, nfl_id, min(time) AS ts
                FROM game_tracking WHERE event = 'ball_snap' GROUP BY ALL
            )
            SELECT t.game_id, t.play_id, t.nfl_id, epoch(t.time) AS t, t.x, t.y, t.s
            FROM game_tracking t JOIN snap USING (game_id, play_id, nfl_id) JOIN games g USING (game_id)
            WHERE g.season = {season} AND g.season_type = 'REG' AND t.time >= snap.ts
            ORDER BY t.nfl_id, t.game_id, t.play_id, t.time
            """
        ).df()
        ids = df["nfl_id"].unique()
        parts = [df[df["nfl_id"].isin(chunk)] for chunk in np.array_split(ids, 16)]
        for rows in Parallel(n_jobs=n_jobs)(delayed(_features)(p) for p in parts):
            out.extend(rows)
    pd.DataFrame(out).to_parquet(artifacts_dir() / "game_play_features.parquet", index=False)
