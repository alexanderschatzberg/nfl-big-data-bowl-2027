"""Three position studies of one chain: trait -> transfer -> value.

1. Trait: a drill-derived movement trait is repeatable and not just the stopwatch.
2. Transfer: that trait reappears in the same player's game tracking.
3. Value: traits that transfer buy context-adjusted results.
Each stage gates the next; the position groups are replications of the chain.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .combine import family_features
from .data import artifacts_dir, connect

STOPWATCH = ["combine_height", "combine_weight", "forty", "ten_yd_split", "vertical", "broad_jump"]
UNDRAFTED_PICK = 260


@dataclass(frozen=True)
class Hypothesis:
    name: str
    title: str
    position: str  # combine_position
    families: dict  # family -> (drills, features)
    transfer: str  # the same movement, measured in game tracking
    value: tuple  # context-adjusted results that movement should buy

    @property
    def outcomes(self) -> tuple:
        return (self.transfer, *self.value)


_SPRINT = (["FORTY_YARD_DASH"], ["f40_a0", "f40_vmax"])

HYPOTHESES = {h.name: h for h in [
    Hypothesis(
        "wr_break", "WR break efficiency -> separation on breaking routes", "WR",
        {
            "stop": (["COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT"],
                     ["brk_v_in", "brk_dec", "brk_stop_eff", "brk_reacc_10", "brk_time_under"]),
            "cut": (["SPEED_OUT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "POST_CORNER_ROUTE_LEFT",
                     "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT", "SLANT_ROUTE_LEFT"],
                    ["curve_speed", "lat_acc", "dec_max"]),
            "go": (["GO_ROUTE_RIGHT"], ["vmax", "accel_2to5"]),
            "sprint": _SPRINT,
        },
        "game_braking_oe", ("sep_oe_break", "sep_oe_all"),
    ),
    Hypothesis(
        "ol_mirror", "OL mirror agility -> pressure allowed", "OL",
        {
            "mirror": (["PASS_PRO_MIRROR_DRILL"], ["cod_rate", "cod_reacc", "dec_max", "acc_max"]),
            "wave": (["FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"], ["cod_rate", "cod_reacc", "cod_peak_speed"]),
            "set": (["PASS_RUSH_DROP"], ["accel_1to3", "vmax", "path_eff"]),
            "pull": (["OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT",
                      "OL_PULL_DRILL_LONG_PULL_RIGHT", "SCREEN_DRILL"], ["accel_2to5", "vmax"]),
            "sprint": _SPRINT,
        },
        "game_set_burst_oe", ("pressure_allowed_oe", "peak_pressure_prob_oe"),
    ),
    Hypothesis(
        "dl_getoff", "DL get-off and bend -> get-off time and pressure", "DL",
        {
            "rush": (["PASS_RUSH_DRILL"], ["accel_2to5", "vmax", "lat_acc"]),
            "bend": (["RUN_THE_HOOP_DRILL"], ["curve_speed", "lat_acc"]),
            "agility": (["FOUR_BAG_AGILITY_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION",
                         "BODY_CONTROL_DRILL", "RUN_AND_CLUB_DRILL"], ["cod_reacc", "dec_max", "acc_max"]),
            "sprint": _SPRINT,
        },
        "game_burst_oe", ("get_off_oe", "pressure_rate_oe"),
    ),
]}


@dataclass
class Design:
    df: pd.DataFrame
    stopwatch: list  # size + official drill numbers + class + position
    draft: list  # draft slot
    tracking: list


def load_design(hyp: Hypothesis) -> Design:
    con = connect()
    meta = con.execute(
        """
        SELECT p.nfl_id, p.display_name, p.draft_year, p.nfl_position, p.draft_overall_pick,
               cr.* EXCLUDE (nfl_id, draft_year)
        FROM players p JOIN combine_results cr USING (nfl_id)
        WHERE cr.combine_position = ?
        """,
        [hyp.position],
    ).df()
    meta["log_pick"] = np.log(meta["draft_overall_pick"].fillna(UNDRAFTED_PICK).astype(float))
    dummies = pd.get_dummies(meta[["draft_year", "nfl_position"]].astype(str), drop_first=True, dtype=float)
    dummies = dummies.loc[:, dummies.sum() >= 5]
    player_drill = pd.read_parquet(artifacts_dir() / "combine_player_drill.parquet")
    track = family_features(player_drill, hyp.families)
    outcomes = pd.read_parquet(artifacts_dir() / f"outcomes_{hyp.position}.parquet")
    df = pd.concat([meta, dummies], axis=1).merge(track, on="nfl_id", how="left").merge(outcomes, on="nfl_id", how="left")
    tracking = [c for c in track.columns if c != "nfl_id"]
    return Design(df, STOPWATCH + list(dummies.columns), ["log_pick"], tracking)
