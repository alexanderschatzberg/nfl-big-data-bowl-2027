"""Do players improve from their first NFL season to a later one, and does the Combine predict who?

Season 1 -> 3 exists only for the 2023 class; season 1 -> 2 adds the 2024
class and is reported alongside as the better-powered version. Two questions
per position and measure:

1. Does the group improve? Paired change among players who qualify in both seasons.
2. Is the improvement predictable? Each Combine predictor against the later
   season with the first season held fixed (so a predictor that merely tracks
   the season-1 level gets no credit), with a min-p permutation test over the
   whole family for that position and horizon.

Rates are centred within position and usual alignment. Playing time counts a season with no snaps as zero. The rate measures need a
minimum number of plays in both seasons, so they describe survivors only.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import evaluate, interactions
from .data import artifacts_dir, connect

# measure -> (count column that must reach the minimum, minimum)
MEASURES = {
    "WR": {"yards_per_route": ("routes", 50), "targets_per_route": ("routes", 50), "separation": ("sep_n", 30)},
    "TE": {"yards_per_route": ("routes", 50), "targets_per_route": ("routes", 50), "separation": ("sep_n", 30)},
    "OL": {"pressure_kept_out": ("pass_blocks", 50)},
    "DL": {"pressure_rate": ("rushes", 50), "get_off": ("rushes", 50)},
    "DB": {},
}
COMMON = {"log_snaps": ("always", 0), "game_speed": ("speed_n", 50)}
MIN_N = 15


def seasons() -> pd.DataFrame:
    """One row per player and NFL season number (1 = rookie year), zero-filled for seasons not played."""
    con = connect()
    s = con.execute(
        """
        SELECT pp.nfl_id, g.season, count(*) AS snaps,
               count(pp.route_ran) AS routes,
               sum(coalesce(pp.rec_yards, 0)) FILTER (pp.route_ran IS NOT NULL) / nullif(count(pp.route_ran), 0) AS yards_per_route,
               avg(pp.target::DOUBLE) FILTER (pp.route_ran IS NOT NULL) AS targets_per_route,
               count(pp.separation_at_pass_forward) AS sep_n, avg(pp.separation_at_pass_forward) AS separation,
               count(pp.pressure_allowed) AS pass_blocks, 1 - avg(pp.pressure_allowed::DOUBLE) AS pressure_kept_out,
               count(pp.player_get_off) AS rushes,
               avg((pp.time_to_pressure IS NOT NULL)::DOUBLE) FILTER (pp.player_get_off IS NOT NULL) AS pressure_rate,
               -avg(pp.player_get_off) AS get_off
        FROM player_play pp JOIN games g USING (game_id)
        WHERE g.season_type = 'REG' AND pp.play_nullified_by_penalty = 'N'
        GROUP BY ALL
        """
    ).df()
    f = pd.read_parquet(artifacts_dir() / "game_play_features.parquet", columns=["game_id", "nfl_id", "vmax"])
    f = f.merge(con.execute("SELECT game_id, season FROM games").df(), on="game_id").groupby(["nfl_id", "season"])["vmax"]
    speed = pd.DataFrame({"game_speed": f.quantile(0.99), "speed_n": f.count()}).reset_index()
    players = con.execute("SELECT nfl_id, draft_year, combine_position AS position FROM combine_results").df()
    grid = players.merge(pd.DataFrame({"season": [2023, 2024, 2025]}), how="cross")
    grid = grid[grid["season"] >= grid["draft_year"]]
    d = grid.merge(s, on=["nfl_id", "season"], how="left").merge(speed, on=["nfl_id", "season"], how="left")
    counts = ["snaps", "routes", "sep_n", "pass_blocks", "rushes", "speed_n"]
    d[counts] = d[counts].fillna(0)
    # Tackles face more pressure than guards and outside receivers run faster routes than slot
    # ones, so every rate is taken relative to the player's usual alignment (see `conventional.frame`).
    role = con.execute("SELECT nfl_id, mode(lined_up_position) AS role FROM player_play GROUP BY nfl_id").df()
    d = d.merge(role, on="nfl_id", how="left")
    d.loc[d.groupby(["position", "role"])["nfl_id"].transform("nunique") < 8, "role"] = "OTHER"
    rates = sorted({m for ms in MEASURES.values() for m in ms} | {"game_speed"})
    d[rates] -= d.groupby(["position", "role"], dropna=False)[rates].transform("mean")
    d["log_snaps"] = np.log1p(d["snaps"])
    d["always"] = 1
    d["year"] = d["season"] - d["draft_year"] + 1
    return d


def pairs(position: str, later: int, s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """measure -> frame of players qualifying in season 1 and season `later`, columns y1, y2."""
    s = s[s["position"] == position]
    a, b = s[s["year"] == 1].set_index("nfl_id"), s[s["year"] == later].set_index("nfl_id")
    out = {}
    for m, (count, minimum) in (MEASURES[position] | COMMON).items():
        d = pd.DataFrame({"y1": a[m].where(a[count] >= minimum), "y2": b[m].where(b[count] >= minimum)}).dropna()
        if len(d) >= MIN_N:
            out[m] = d
    return out


def group_change(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for later in (2, 3):
        for position in MEASURES:
            for m, d in pairs(position, later, s).items():
                diff = d["y2"] - d["y1"]
                rows.append({"position": position, "horizon": f"1->{later}", "measure": m, "n": len(d),
                             "season1": d["y1"].mean(), "later": d["y2"].mean(),
                             "change_sd": diff.mean() / d["y1"].std(), "improved": (diff > 0).mean(),
                             "p": stats.wilcoxon(diff).pvalue, "stability": stats.spearmanr(d["y1"], d["y2"])[0]})
    return pd.DataFrame(rows)


def _scan(P: pd.DataFrame, frames: dict, perm: np.ndarray | None = None) -> list[dict]:
    if perm is not None:
        P = pd.DataFrame(P.to_numpy(float)[perm], index=P.index, columns=P.columns)
    rows = []
    for m, d in frames.items():
        X = P.reindex(d.index)
        for c in X.columns:
            ok = X[c].notna().to_numpy()
            if ok.sum() < MIN_N or np.ptp(X[c].to_numpy()[ok]) == 0:
                continue
            x, y1, y2 = X[c].to_numpy()[ok], d["y1"].to_numpy()[ok], d["y2"].to_numpy()[ok]
            Z = y1[:, None]
            rho, p = stats.spearmanr(evaluate._residualize(x, Z), evaluate._residualize(y2, Z))
            rows.append({"measure": m, "predictor": c, "n": int(ok.sum()), "partial_rho": rho, "p": p,
                         "rho_with_change": stats.spearmanr(x, y2 - y1)[0], "rho_with_season1": stats.spearmanr(x, y1)[0]})
    return rows


def predict_change(s: pd.DataFrame, n_perm: int = 2000) -> pd.DataFrame:
    tables = []
    for later in (2, 3):
        for position in MEASURES:
            frames = pairs(position, later, s)
            if not frames:
                continue
            d = interactions.design(position)
            P = d["df"].set_index("nfl_id")[[c for c in dict.fromkeys(x for a, b, *_ in d["pairs"] for x in (a, b))
                                             if not c.startswith("is_")]]
            t = pd.DataFrame(_scan(P, frames))
            if t.empty:
                continue
            rng = np.random.default_rng(0)
            null = np.sort([min(r["p"] for r in _scan(P, frames, rng.permutation(len(P)))) for _ in range(n_perm)])
            t["p_family"] = (1 + np.searchsorted(null, t["p"].to_numpy(), side="right")) / (1 + n_perm)
            t.insert(0, "horizon", f"1->{later}")
            t.insert(0, "position", position)
            tables.append(t)
    return pd.concat(tables).sort_values("p")
