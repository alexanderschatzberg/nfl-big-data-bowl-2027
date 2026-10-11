"""Every number and table the writeup quotes, computed in one place.

The notebook calls these functions and `figures.py` draws from their results,
so a number in the writeup can be traced to one function here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from . import conventional, development, interactions, matched, usage, value

HEADLINE = "stop-route braking"
SPEED = "40 tracked top speed"
STOPWATCH_OUTCOMES = {"game_speed_p99": "Top speed in games", "sep_oe_all": "Separation from the defender",
                      "targets_per_route": "Targets per route", "yards_per_route": "Yards per route"}


def _parallel_null(fn, position: str, n_perm: int) -> list:
    return Parallel(n_jobs=-1)(delayed(fn)(position, list(range(i, min(i + 25, n_perm))))
                               for i in range(0, n_perm, 25))


def stopwatch(n_perm: int = 1000) -> pd.DataFrame:
    """Conventional predictors and the eleven pre-specified interactions, all positions."""
    tables = []
    for pos in conventional.POSITIONS:
        null = [x for chunk in _parallel_null(conventional.min_p_null, pos, n_perm) for x in chunk]
        tables.append(conventional.adjust(conventional.run(pos), null))
    return pd.concat(tables, ignore_index=True)


def interaction_scan(n_perm: int = 1000) -> pd.DataFrame:
    tables = [interactions.adjust(interactions.scan(pos), np.vstack(_parallel_null(interactions.min_p_null, pos, n_perm)))
              for pos in interactions.TRAITS]
    return pd.concat(tables, ignore_index=True)


def _boot(d: pd.DataFrame, n_boot: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    draws = [matched.test(d.iloc[rng.integers(0, len(d), len(d))])["partial_rho"] for _ in range(n_boot)]
    return np.nanpercentile(draws, [2.5, 97.5]).tolist()


def chain(n_boot: int = 2000) -> pd.DataFrame:
    """Trait -> transfer -> value for receivers: partial correlations with bootstrap intervals."""
    plays = matched.game_plays()
    rows = []

    def add(stage, label, d, primary=False):
        t = matched.test(d)
        lo, hi = _boot(d, n_boot)
        rows.append({"stage": stage, "label": label, "n": t["n"], "partial_rho": t["partial_rho"],
                     "p": t["partial_p"], "lo": lo, "hi": hi, "pre_specified": primary})

    transfer = {HEADLINE: "Braking in the comeback and curl drills, and on hitch routes",
                SPEED: "Top speed in the 40-yard dash, and on go routes"}
    for name, label in transfer.items():
        add("Combine movement and the same movement in games", label, matched.pair_frame(next(p for p in matched.PAIRS if p.name == name), plays))
    outcome = {("separation", "MAN_COVERAGE"): "separation against man coverage", ("separation", "ZONE_COVERAGE"): "separation against zone coverage",
               ("target", "MAN_COVERAGE"): "target rate against man coverage", ("yac", None): "yards after the catch, compared with expected"}
    stage = {"combine braking": ("Combine braking and results on hitch routes", "Stop-drill braking"),
             "game hitch braking": ("Game braking and results on hitch routes", "Game hitch braking")}
    for i, (pred, y, cov, d) in enumerate(value.frames()):
        add(stage[pred][0], f"{stage[pred][1]}: {outcome[(y, cov)]}", d, primary=i == 0)
    return pd.DataFrame(rows)


def scatter() -> pd.DataFrame:
    """One row per receiver: drill braking, game hitch braking, draft class."""
    pair = next(p for p in matched.PAIRS if p.name == HEADLINE)
    return matched.pair_frame(pair, matched.game_plays())


def stopwatch_frame() -> pd.DataFrame:
    return conventional.frame("WR")[["nfl_id", "forty", "combine_weight"] + list(STOPWATCH_OUTCOMES)]


def test_count(tables: dict[str, pd.DataFrame]) -> dict:
    counts = {k: int(len(v)) for k, v in tables.items()}
    return counts | {"total": int(sum(counts.values()))}


def _row(t: pd.DataFrame, **where) -> pd.Series:
    m = np.ones(len(t), bool)
    for k, v in where.items():
        m &= (t[k].isna() if v is None else t[k] == v).to_numpy()
    return t[m].iloc[0]


def numbers(tables: dict, robustness: pd.DataFrame, benchmarks: dict, chain_table: pd.DataFrame,
            group: pd.DataFrame, retention: pd.DataFrame, game_movement: pd.DataFrame,
            coverage: pd.DataFrame) -> dict:
    """The quoted values, rounded as the writeup prints them."""
    sw, mt, it, us, dv = (tables[k] for k in ("conventional", "matched", "interactions", "usage", "development"))
    wr = sw[(sw["position"] == "WR") & (sw["kind"] == "main")]
    out = {"stopwatch_wr": {}}
    for pred in ("forty", "combine_weight", "combine_height", "log_pick"):
        for y in STOPWATCH_OUTCOMES:
            r = _row(wr, a=pred, outcome=y)
            out["stopwatch_wr"][f"{pred}->{y}"] = {"n": int(r["n"]), "rho": round(r["effect"], 2), "p": round(r["p"], 4),
                                                   "p_family": round(r["p_family"], 3)}
    for name in (HEADLINE, SPEED):
        r = _row(mt, trait=name)
        out[name] = {"n": int(r["n"]), "rho": round(r["rho"], 2), "partial_rho": round(r["partial_rho"], 2),
                     "partial_p": float(f"{r['partial_p']:.2g}"), "partial_q": round(r["partial_q"], 3),
                     "by_class": r["by_class"]}
    out["matched_family"] = {"tests": len(mt), "survive_q05": mt.loc[mt["partial_q"] < 0.05, "trait"].tolist()}
    out[HEADLINE] |= {"ci95": robustness.attrs["ci95"], "perm_p": round(float(robustness.attrs["perm_p"]), 4)}
    out["robustness"] = {r["variant"]: round(r["partial_rho"], 2) for _, r in robustness.iterrows()}
    out["benchmark"] = {k: {"combine_to_game": round(float(b["combine_to_game"]), 2), "combine_ci": np.round(b["combine_ci"], 2).tolist(),
                            "n": int(b["n"]), "season_to_next_season": round(float(b["season_to_next_season"]), 2),
                            "season_ci": np.round(b["season_ci"], 2).tolist(), "season_pairs": int(b["season_pairs"]),
                            "season_players": int(b["season_players"])} for k, b in benchmarks.items()}
    out["chain"] = [{"label": r["label"], "n": int(r["n"]), "partial_rho": round(r["partial_rho"], 2), "p": round(r["p"], 3),
                     "ci": [round(r["lo"], 2), round(r["hi"], 2)], "pre_specified": bool(r["pre_specified"])}
                    for _, r in chain_table.iterrows()]
    out["game_movement"] = {"tests": len(game_movement), "q_below_10": int((game_movement["q"] < 0.10).sum()),
                            "separation_reliability": round(float(game_movement.attrs["separation_reliability"]), 2),
                            "top": game_movement.head(3).round(3).to_dict("records")}
    out["coverage_profile"] = {"tests": len(coverage), "max_abs_partial": round(float(coverage["partial_rho"].abs().max()), 2),
                               "min_q": round(float(coverage["q"].min()), 2),
                               **{k: round(v, 2) if isinstance(v, float) else v for k, v in coverage.attrs.items()}}
    out["usage"] = {"tests": len(us), "min_partial_q": round(float(us["partial_q"].min()), 2)}
    out["interactions"] = {"tests": len(it), "p_family_below_05": int((it["p_family"] < 0.05).sum()),
                           "top": it.sort_values("p_family").head(3)[["position", "a", "b", "outcome", "n", "beta", "p_family"]]
                           .round(3).to_dict("records")}
    out["development"] = {"tests": len(dv), "min_p_family": round(float(dv["p_family"].min()), 3),
                          "top": dv.sort_values("p").head(3).round(3).to_dict("records"),
                          "group": group.round(3).to_dict("records"), "dl_retention": retention.round(3).to_dict("records")}
    out["tests"] = test_count(tables | {"robustness": robustness, "value": chain_table.iloc[2:],
                                        "game_movement": game_movement, "coverage_profile": coverage,
                                        "development_group": group})
    return out


def save(numbers_: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(numbers_, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
