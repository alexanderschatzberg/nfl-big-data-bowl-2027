# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

This is a Kaggle NFL Big Data Bowl 2027 entry ("Combine to NFL Performance"). As of the initial commit there is **no code, no environment, and no build/test tooling** — only the competition brief (`instructions.md`) and the raw data in `data/`. `python3` and `uv` are available locally; R is not installed. When adding the first code, also record the chosen run/test commands here.

`data/*` is gitignored (≈2.3 GB). Never commit it, and don't assume it exists in a fresh clone.

## The deliverable

Not a predictive-model leaderboard — a judged writeup (see `instructions.md` for the full rubric):

- ≤ 2,000 words and fewer than 10 tables/figures; over-limit submissions are penalized.
- Must explicitly link **Combine sensor tracking** to **regular-season NFL game performance**. A writeup that doesn't use the tracking data is not scored.
- Must have an attached public Kaggle Notebook; most code belongs in an appendix/notebook, not the writeup body.
- Scored on Football (30%), Data Science (30%), Writeup (20%), Data Visualization (20%) by NFL team analytics staff. The brief explicitly encourages a narrow focus (one position group, one drill, one aspect of movement) over broad coverage.
- Final submission deadline: January 6, 2027, 11:59 PM UTC.

## Data model

Everything joins on `nfl_id`. The cohort is 510 players from the 2023–2025 draft classes; `players.csv`, `combine_results.csv`, and `player_career_successes.csv` each have exactly one row per cohort player.

| File | Grain | Notes |
|---|---|---|
| `players.csv` | player | `draft_year`, `nfl_position` (fine-grained: WR, CB, G, T, DT, DE, OLB, SS, FS, …), draft slot |
| `combine_results.csv` | player | Stopwatch/measurement numbers (`forty`, `ten_yd_split`, `three_cone`, `short_shuttle`, …) plus NGS scores. `combine_position` is coarse: DB, OL, DL, WR, TE only. Many `NA`s — players skip drills |
| `combine_tracking.csv` | 10 Hz frame | 463k rows. Key: `nfl_id` + `event_id` + `drill_name` + `attempt` (~6.3k player attempts, ~68 frames each) |
| `game_tracking_{2023,2024,2025}.csv` | 10 Hz frame | 22.6M rows total, ~2 GB. Key: `game_id` + `play_id` + `nfl_id` |
| `player_play.csv` | player-play | 327k rows, 62 columns: play context (down, EPA, WP, formation, coverage) plus per-player outcomes |
| `games.csv` | game | `season`, `season_type` (REG / PRE / POST), `week` |
| `player_career_successes.csv` | player | Career snaps, starts, All-Pro, Pro Bowl |

Things that are not obvious from the headers:

- **Game tracking covers only cohort players, not all 22 on the field.** Most plays have 1–3 tracked players and there is no ball row. Anything needing opponent or ball position (e.g. separation, cushion, pressure) must come from the precomputed columns in `player_play.csv`, not be derived from tracking.
- **Combine tracking has `BALL` rows** (`entity_type`), keyed to the `nfl_id` of the player running the drill. Filter to `entity_type == "PLAYER"` for movement metrics.
- **Combine tracking has no `o` (orientation) column**; game tracking has both `o` and `dir`.
- **Drill names are inconsistent across years.** The same drill appears under variants such as `PASS_PRO-MIRROR_DRILL` / `PASS_PRO_MIRROR_DRILL`, `OVER_SHOULDER_ADJUST` / `OVER_THE_SHOULDER_ADJUST`, `SPEED_OUT_LEFT` / `SPEED_OUT_ROUTE_LEFT`, `LINE` / `LINE_DRILL`. Normalize before grouping.
- **Drill coverage is by position group.** `drill_type` is one of `FORTY_YARD_DASH`, `SHORT_SHUTTLE`, `THREE_CONE_DRILL`, or `SKILL_DRILLS_{DB,DL,LB,OL,TE,WR}`. `SKILL_DRILLS_LB` is tiny (~3k frames); there are no QB or RB drills (the cohort has effectively no QBs/RBs).
- **Game tracking files are split by season, not calendar year** — `game_tracking_2023.csv` includes January 2024 games. Use `games.csv` for season/week, not the `game_id` prefix.
- **`games.csv` includes preseason and postseason.** The brief is about regular-season performance, so filter `season_type == "REG"` deliberately.
- **Exposure is heavily unbalanced by draft class.** The 2023 class has up to three seasons of game data, 2025 has one; `player_play.csv` covers 497 of the 510 players. Account for this (and for draft position / opportunity) before comparing outcomes across classes.
- `player_play.csv` outcome columns are role-specific and `NA` elsewhere: receivers (`route_ran`, `separation_at_pass_forward`, `cushion`, `yards_after_catch`), pass blockers (`pressure_allowed`, `peak_pressure_probability_allowed`, `time_to_pressure_allowed`), pass rushers (`player_get_off`, `time_to_pressure`, `quick_pressure`), coverage (`coverage_assignment`, `team_coverage_man_zone`). Note the source typo in `visitor_team_win_probility_added`.
- CSVs use the literal string `NA` for missing values.
- The game tracking `event` column is `NA` on most frames; tagged frames include `ball_snap`, `pass_forward`, `pass_arrived`, `first_contact`, `tackle`, `handoff`.

## Working with the data

The game tracking files are too large to load casually (the 2025 file alone is 1 GB / 11M rows). Filter by `nfl_id` or `game_id`/`play_id` while reading, or convert to a columnar format once and query that, rather than reading whole CSVs into memory.
