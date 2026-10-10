"""Does Combine braking pay off against man coverage? A pre-specified test.

Stop-drill braking transfers to hitch routes (`matched.py`) but showed no link
to separation when all hitches were pooled. Against zone the defender sits on
the route however the receiver stops, so the pooled test may have hidden a
man-coverage effect. This was written, and committed, before it was run:

PRIMARY (one test, two-sided, alpha 0.05)
  Players    WRs with stop-drill braking and >= 8 qualifying plays.
  Predictor  mean class-relative z of `dec_win05` over the comeback and curl drills.
  Plays      regular season, not nullified, HITCH route, MAN_COVERAGE, separation recorded.
  Outcome    player mean of separation at pass forward, after removing the linear
             effect of cushion at play level.
  Statistic  Spearman partial correlation net of 40 time, weight, height and
             modal alignment (`matched.test`). Expected sign: positive.

SECONDARY (reported, not confirmatory)
  The same against ZONE_COVERAGE; target rate on hitches against man (>= 15
  routes); yards after catch over expected on hitch catches, any coverage
  (>= 5 catches); and each with in-game hitch braking (p90) as the predictor.
"""
import numpy as np
import pandas as pd

from . import matched
from .data import artifacts_dir, connect

MIN_PLAYS = {"separation": 8, "target": 15, "yac": 5}


def hitches(position: str = "WR") -> pd.DataFrame:
    return connect().execute(
        """
        SELECT pp.nfl_id, pp.game_id, pp.play_id, pp.team_coverage_man_zone AS coverage, pp.cushion,
               pp.separation_at_pass_forward AS separation, pp.target::DOUBLE AS target,
               pp.yards_after_catch - pp.expected_yards_after_catch AS yac
        FROM player_play pp JOIN games g USING (game_id) JOIN combine_results cr USING (nfl_id)
        WHERE g.season_type = 'REG' AND pp.play_nullified_by_penalty = 'N'
          AND cr.combine_position = ? AND pp.route_ran = 'HITCH'
        """,
        [position],
    ).df()


def _outcome(h: pd.DataFrame, y: str, coverage: str | None) -> pd.Series:
    d = h if coverage is None else h[h["coverage"] == coverage]
    d = d.dropna(subset=[y]).copy()
    if y == "separation":
        cushion = d["cushion"].fillna(d["cushion"].median())
        d[y] -= np.polyval(np.polyfit(cushion, d[y], 1), cushion)
    g = d.groupby("nfl_id")[y]
    return g.mean().where(g.count() >= MIN_PLAYS[y]).dropna().rename("g")


TESTS = [("separation", "MAN_COVERAGE"), ("separation", "ZONE_COVERAGE"), ("target", "MAN_COVERAGE"), ("yac", None)]


def frames(position: str = "WR") -> list[tuple]:
    """(predictor, outcome, coverage, frame) for every test; the first is the primary."""
    h = hitches(position)
    plays = matched.game_plays()
    pair = next(p for p in matched.PAIRS if p.name == "stop-route braking")
    base = matched.pair_frame(pair, plays, min_plays=0).drop(columns="g")
    game = matched.pair_frame(pair, plays).set_index("nfl_id")["g"].rename("c")
    predictors = {"combine braking": base, "game hitch braking": base.drop(columns="c").merge(game, on="nfl_id")}
    return [(name, y, cov, frame.merge(_outcome(h, y, cov), on="nfl_id"))
            for name, frame in predictors.items() for y, cov in TESTS]


def table(position: str = "WR") -> pd.DataFrame:
    rows = [{"predictor": name, "outcome": y, "coverage": cov or "any",
             "primary": name == "combine braking" and (y, cov) == TESTS[0]} | matched.test(d)
            for name, y, cov, d in frames(position)]
    return pd.DataFrame(rows)


ROUTE_SETS = {"hitch": ["HITCH"], "cuts": ["OUT", "IN", "SLANT", "CORNER", "POST"], "go": ["GO"], "all": None}
GAME_FEATURES = ["vmax", "dec_win05", "acc_win05", "lat_acc", "cut_turn_rate", "curve_speed", "brk_reacc_10", "path_eff"]


def game_movement(position: str = "WR", quantile: float = 0.9, min_plays: int = 20) -> pd.DataFrame:
    """Exploratory: does movement capability measured *in games* relate to separation or yards?

    If it does not, a null from the Combine side says little about the Combine:
    the outcome is not driven by movement capability wherever it is measured.
    `attrs["separation_reliability"]` shows the outcome is not simply noise.
    """
    from scipy import stats

    from . import evaluate

    con = connect()
    out = con.execute(
        """
        SELECT game_id, play_id, nfl_id, separation_at_pass_forward AS sep, cushion AS cush,
               coalesce(rec_yards, 0) AS yds
        FROM player_play
        """
    ).df()
    w = matched.game_plays().drop(columns="cushion")
    w = w[(w["position"] == position) & w["route_ran"].notna()].merge(out, on=["game_id", "play_id", "nfl_id"])
    w["cush"] = w["cush"].fillna(w["cush"].median())
    s = w.dropna(subset=["sep"]).copy()
    s["sep"] -= s.groupby("route_ran")["sep"].transform("mean")
    s["sep"] -= np.polyval(np.polyfit(s["cush"], s["sep"], 1), s["cush"])
    meta = con.execute(f"SELECT nfl_id, {', '.join(matched.SIZE_SPEED)} FROM combine_results").df().set_index("nfl_id")
    role = w.groupby("nfl_id")["lined_up"].agg(lambda x: x.mode().iat[0]).rename("role")
    rows = []
    for name, routes in ROUTE_SETS.items():
        a = w if routes is None else w[w["route_ran"].isin(routes)]
        b = s if routes is None else s[s["route_ran"].isin(routes)]
        g, h = a.groupby("nfl_id"), b.groupby("nfl_id")
        outcomes = {"separation": h["sep"].mean().where(h.size() >= min_plays),
                    "yards per route": g["yds"].mean().where(g.size() >= 2 * min_plays)}
        for f in GAME_FEATURES:
            x = g[f].quantile(quantile).where(g[f].count() >= min_plays).rename("x")
            for oname, y in outcomes.items():
                d = pd.concat([x, y.rename("y"), meta, role], axis=1).dropna(subset=["x", "y"])
                Z = np.column_stack([d[matched.SIZE_SPEED].fillna(d[matched.SIZE_SPEED].median()).to_numpy(float),
                                     pd.get_dummies(d["role"], drop_first=True, dtype=float).to_numpy()])
                rho, p = stats.spearmanr(evaluate._residualize(d["x"].to_numpy(float), Z),
                                         evaluate._residualize(d["y"].to_numpy(float), Z))
                rows.append({"routes": name, "game_feature": f, "outcome": oname, "n": len(d), "partial_rho": rho, "p": p})
    t = pd.DataFrame(rows)
    t["q"] = stats.false_discovery_control(t["p"], method="bh")
    rng = np.random.default_rng(0)
    n = s.groupby("nfl_id").size()
    keep = n[n >= 2 * min_plays].index
    halves = []
    for _ in range(100):
        half = rng.integers(0, 2, len(s))
        a, b = (s[half == k].groupby("nfl_id")["sep"].mean().reindex(keep) for k in (0, 1))
        halves.append(stats.spearmanr(a, b)[0])
    r = float(np.mean(halves))
    t.attrs["separation_reliability"] = 2 * r / (1 + r)
    return t.sort_values("p")


COVERAGES = {"Man (all)": "mz == 'MAN_COVERAGE'", "Zone (all)": "mz == 'ZONE_COVERAGE'",
             "Cover 1": "ctype == 'COVER_1_MAN'", "Cover 2": "ctype == 'COVER_2_ZONE'",
             "Cover 3": "ctype == 'COVER_3_ZONE'", "Cover 4 or 6": "ctype in ['COVER_4_ZONE', 'COVER_6_ZONE']"}
PROFILE = {"braking": (matched._STOP, "dec_win05"), "top speed": (["FORTY_YARD_DASH"], "vmax")}


def coverage_profile(position: str = "WR", min_plays: int = 25) -> pd.DataFrame:
    """Exploratory: do the two Combine traits that transfer relate to separation against particular coverages?

    Separation is adjusted for route and cushion at play level. Partial correlations
    remove weight, height and alignment. `attrs` holds how related the two traits
    are to each other, and how related a receiver's man and zone separation are.
    """
    from scipy import stats

    from . import evaluate

    con = connect()
    pl = con.execute(
        """
        SELECT pp.nfl_id, pp.route_ran AS route, pp.team_coverage_type AS ctype, pp.team_coverage_man_zone AS mz,
               pp.separation_at_pass_forward AS sep, pp.cushion, pp.lined_up_position AS lined_up
        FROM player_play pp JOIN games g USING (game_id) JOIN combine_results cr USING (nfl_id)
        WHERE g.season_type = 'REG' AND pp.play_nullified_by_penalty = 'N' AND cr.combine_position = ?
          AND pp.route_ran IS NOT NULL AND pp.separation_at_pass_forward IS NOT NULL
        ORDER BY pp.game_id, pp.play_id, pp.nfl_id
        """,
        [position],
    ).df()
    pl["cushion"] = pl["cushion"].fillna(pl["cushion"].median())
    pl["sep"] -= pl.groupby("route")["sep"].transform("mean")
    pl["sep"] -= np.polyval(np.polyfit(pl["cushion"], pl["sep"], 1), pl["cushion"])
    drills = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    drills = drills[drills["combine_position"] == position]
    traits = pd.concat([drills[drills["drill"].isin(d)].groupby("nfl_id")[f].mean().rename(name)
                        for name, (d, f) in PROFILE.items()], axis=1)
    size = ["combine_weight", "combine_height"]
    meta = con.execute(f"SELECT nfl_id, {', '.join(size)} FROM combine_results").df().set_index("nfl_id")
    role = pl.groupby("nfl_id")["lined_up"].agg(lambda s: s.mode().iat[0]).rename("role")

    def player_mean(where: str) -> pd.Series:
        g = pl.query(where).groupby("nfl_id")["sep"]
        return g.mean().where(g.count() >= min_plays).dropna()

    rows = []
    for coverage, where in COVERAGES.items():
        y = player_mean(where).rename("y")
        for trait in traits:
            d = pd.concat([traits[trait].rename("x"), y, meta, role], axis=1).dropna(subset=["x", "y"])
            Z = np.column_stack([d[size].fillna(d[size].median()).to_numpy(float),
                                 pd.get_dummies(d["role"], drop_first=True, dtype=float).to_numpy()])
            rho, p = stats.spearmanr(evaluate._residualize(d["x"].to_numpy(float), Z),
                                     evaluate._residualize(d["y"].to_numpy(float), Z))
            rows.append({"coverage": coverage, "trait": trait, "n": len(d), "partial_rho": rho, "p": p})
    t = pd.DataFrame(rows)
    t["q"] = stats.false_discovery_control(t["p"], method="bh")
    both = traits.dropna()
    mz = pd.concat([player_mean(COVERAGES["Man (all)"]), player_mean(COVERAGES["Zone (all)"])], axis=1).dropna()
    t.attrs = {"traits_rho": float(stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]), "traits_n": len(both),
               "man_zone_rho": float(stats.spearmanr(mz.iloc[:, 0], mz.iloc[:, 1])[0]), "man_zone_n": len(mz)}
    return t
