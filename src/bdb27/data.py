"""Paths, CSV -> Parquet conversion and a DuckDB connection over the tables."""
import os
from pathlib import Path

import duckdb

SMALL = ["players", "combine_results", "player_career_successes", "games", "player_play", "combine_tracking"]
GAME_TRACKING = ["game_tracking_2023", "game_tracking_2024", "game_tracking_2025"]


def data_dir() -> Path:
    return Path(os.environ.get("BDB27_DATA", "data"))


def parquet_dir() -> Path:
    return data_dir() / "parquet"


def artifacts_dir() -> Path:
    d = data_dir() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_parquet(force: bool = False) -> None:
    """Convert the raw CSVs once; everything downstream reads Parquet."""
    out = parquet_dir()
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for name in SMALL + GAME_TRACKING:
        dst = out / f"{name}.parquet"
        if dst.exists() and not force:
            continue
        # Sparse role-specific columns need a full scan to be typed correctly.
        sample = "" if name in GAME_TRACKING else ", sample_size=-1"
        tmp = dst.with_suffix(".tmp")
        con.execute(
            f"COPY (SELECT * FROM read_csv('{data_dir() / (name + '.csv')}', nullstr='NA'{sample})) "
            f"TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)"
        )
        tmp.rename(dst)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    pq = parquet_dir()
    for name in SMALL:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{pq / (name + '.parquet')}')")
    con.execute(f"CREATE VIEW game_tracking AS SELECT * FROM read_parquet('{pq}/game_tracking_*.parquet')")
    return con
