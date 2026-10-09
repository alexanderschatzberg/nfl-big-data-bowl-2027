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


def table(position: str = "WR") -> pd.DataFrame:
    h = hitches(position)
    plays = matched.game_plays()
    pair = next(p for p in matched.PAIRS if p.name == "stop-route braking")
    base = matched.pair_frame(pair, plays, min_plays=0).drop(columns="g")
    game = matched.pair_frame(pair, plays).set_index("nfl_id")["g"].rename("c")
    predictors = {"combine braking": base, "game hitch braking": base.drop(columns="c").merge(game, on="nfl_id")}
    tests = [("separation", "MAN_COVERAGE"), ("separation", "ZONE_COVERAGE"), ("target", "MAN_COVERAGE"),
             ("yac", None)]
    rows = []
    for name, frame in predictors.items():
        for y, cov in tests:
            d = frame.merge(_outcome(h, y, cov), on="nfl_id")
            rows.append({"predictor": name, "outcome": y, "coverage": cov or "any",
                         "primary": name == "combine braking" and (y, cov) == tests[0]} | matched.test(d))
    return pd.DataFrame(rows)
