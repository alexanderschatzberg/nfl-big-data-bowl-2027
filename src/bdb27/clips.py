"""Fixed-length, orientation-free movement clips for the fingerprint network.

Both domains go through the same code: a Combine attempt and a post-snap game
play each become a [T, C] array of intrinsic (Frenet-style) channels that do
not depend on where on the field, or in which direction, the player moved.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .combine import DRILL_ALIASES
from .data import artifacts_dir, connect

HZ = 10.0
T_COMBINE = 100
T_GAME = 50
MIN_FRAMES = 30
MAX_GAME_PLAYS = 400  # per player
CHANNELS = ["speed", "tangential_acc", "turn_rate", "lateral_acc", "valid"]
SCALE = np.array([10.0, 5.0, 3.0, 8.0])  # yd/s, yd/s^2, rad/s, yd/s^2
MOVING = 1.0  # yd/s; heading is noise below this


def batch_channels(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[N, T] position/speed arrays (NaN after a clip ends) -> [N, T, C] float32."""
    valid = ~np.isnan(s)
    fill = lambda a: pd.DataFrame(a).ffill(axis=1).bfill(axis=1).to_numpy()  # noqa: E731
    x, y, s = (savgol_filter(fill(a), 5, 2, axis=1) for a in (x, y, s))
    acc = np.gradient(s, 1 / HZ, axis=1)
    heading = np.unwrap(np.arctan2(np.gradient(y, axis=1), np.gradient(x, axis=1)), axis=1)
    omega = np.gradient(heading, 1 / HZ, axis=1) * (s > MOVING)
    out = np.stack([s, acc, omega, s * omega], axis=-1) / SCALE
    out = np.clip(out, -3, 3) * valid[..., None]
    return np.concatenate([out, valid[..., None]], axis=-1).astype(np.float32)


def _to_arrays(df: pd.DataFrame, key: list[str], length: int):
    """Long frames (sorted within clip, with integer `frame`) -> padded [N, T] arrays."""
    df = df[df["frame"] < length]
    clip = df.groupby(key, sort=False).ngroup().to_numpy()
    frame = df["frame"].to_numpy()
    n = clip.max() + 1
    arrays = []
    for col in ("x", "y", "s"):
        a = np.full((n, length), np.nan)
        a[clip, frame] = df[col].to_numpy(float)
        arrays.append(a)
    meta = df.groupby(key, sort=False).agg(nfl_id=("nfl_id", "first"), token=("token", "first"),
                                           n=("frame", "size")).reset_index(drop=True)
    keep = (meta["n"] >= MIN_FRAMES).to_numpy()
    return [a[keep] for a in arrays], meta[keep].reset_index(drop=True)


def build_combine() -> None:
    df = connect().execute(
        """
        SELECT nfl_id, event_id, drill_name, attempt, x, y, s,
               row_number() OVER (PARTITION BY nfl_id, event_id, drill_name, attempt ORDER BY time) - 1 AS frame
        FROM combine_tracking WHERE entity_type = 'PLAYER'
        ORDER BY nfl_id, event_id, drill_name, attempt, time
        """
    ).df()
    df["token"] = df["drill_name"].replace(DRILL_ALIASES)
    (x, y, s), meta = _to_arrays(df, ["nfl_id", "event_id", "drill_name", "attempt"], T_COMBINE)
    _save("combine", batch_channels(x, y, s), meta)


def build_game() -> None:
    con = connect()
    xs, metas = [], []
    for season in (2023, 2024, 2025):  # one season at a time to bound memory
        df = con.execute(
            f"""
            WITH snap AS (
                SELECT game_id, play_id, nfl_id, min(time) AS ts
                FROM game_tracking WHERE event = 'ball_snap' GROUP BY ALL
            ), picked AS (
                SELECT s.*, pp.lined_up_position AS token
                FROM snap s JOIN games g USING (game_id) JOIN player_play pp USING (game_id, play_id, nfl_id)
                WHERE g.season = {season} AND pp.lined_up_position IS NOT NULL
                QUALIFY row_number() OVER (PARTITION BY s.nfl_id ORDER BY hash(s.game_id, s.play_id))
                        <= {MAX_GAME_PLAYS}
            )
            SELECT t.nfl_id, t.game_id, t.play_id, p.token, t.x, t.y, t.s,
                   CAST(round(epoch(t.time - p.ts) * {HZ}) AS INT) AS frame
            FROM game_tracking t JOIN picked p USING (game_id, play_id, nfl_id)
            WHERE t.time >= p.ts AND t.time < p.ts + INTERVAL '{T_GAME / HZ} seconds'
            ORDER BY t.nfl_id, t.game_id, t.play_id, t.time
            """
        ).df()
        (x, y, s), meta = _to_arrays(df, ["nfl_id", "game_id", "play_id"], T_GAME)
        xs.append(batch_channels(x, y, s))
        metas.append(meta)
    _save("game", np.concatenate(xs), pd.concat(metas, ignore_index=True))


def _save(name: str, clips: np.ndarray, meta: pd.DataFrame) -> None:
    vocab = sorted(meta["token"].unique())
    np.savez_compressed(
        artifacts_dir() / f"clips_{name}.npz", clips=clips, nfl_id=meta["nfl_id"].to_numpy(),
        token=meta["token"].map({v: i for i, v in enumerate(vocab)}).to_numpy(), vocab=np.array(vocab),
    )


def build() -> None:
    build_combine()
    build_game()
