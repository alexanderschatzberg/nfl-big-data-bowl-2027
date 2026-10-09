# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra nn --extra modal      # torch and modal are optional extras
uv run pytest                          # all tests
uv run pytest tests/test_kinematics.py::test_stop_and_go_break_is_measured   # one test
uv run bdb27 prep                      # CSV -> Parquet, Combine features, NFL outcomes, clip tensors (~1 min)
uv run bdb27 matched                   # matched drill-to-game transfer table (headline analysis)
uv run bdb27 conventional              # stopwatch predictors + interactions vs success, family-wise p
uv run bdb27 value                     # pre-specified WR braking vs man-coverage separation test
uv run bdb27 usage                     # drill traits vs route / alignment shares
uv run bdb27 development               # season 1->2 and 1->3 change, and Combine predictors of it
uv run bdb27 interactions              # all-pairs interaction scan (~4,600 tests), family-wise p per position
uv run bdb27 screen                    # cheap trait screens, writes outputs/screen_*.csv
uv run bdb27 fit --n-perm 200          # transfer + value tests with permutation nulls
uv run bdb27 fingerprint --folds 0 --steps 300   # local smoke test of the neural model
uv run modal run modal_app.py::main --n-perm 2000 --seeds 3   # full run on T4 GPUs (needs a payment method on Modal)
BDB27_GPU= uv run modal run modal_app.py::main --n-perm 2000 --seeds 3   # same, CPU only
uv run modal run modal_app.py::signature --seeds 3   # Combine-only signature network
```

`BDB27_DATA` overrides the data directory (default `./data`; `/vol` on Modal). `data/*` and `outputs/` are gitignored — never commit them, and don't assume they exist in a fresh clone. R is not installed.

## Architecture

The analysis is one chain tested in three position groups (WR, OL, DL), defined in `src/bdb27/hypotheses.py`. Each stage gates the next:

1. **Trait** — a drill-derived movement trait is repeatable (ICC across attempts/drills) and not predictable from stopwatch numbers. `screen.py`; the gate never looks at an outcome.
2. **Transfer** — the trait reappears in the same player's game tracking. Tested per trait (`fit`, transfer outcome) and by the neural model (`fingerprint.py`).
3. **Value** — traits predict context-adjusted results (separation, pressure, get-off) beyond stopwatch + draft slot.

Data flow, all cached under `data/artifacts/`:

- `data.py` converts CSVs to Parquet once and exposes DuckDB views. All large-table work goes through DuckDB.
- `kinematics.py` turns one 10 Hz trace into per-attempt features (pure numpy; this is what the tests cover). `combine.py` takes the median over attempts and z-scores **within drill × position group × draft class**, because drill setups change between years. Features are combined into `family__feature` columns (mean z across the drills of a family).
- `outcomes.py` builds play-level "over expected" values with gradient boosting **cross-fitted by player** (a player's plays never train their own expectation), then shrinks player means by empirical-Bayes reliability. Every outcome is oriented so higher is better; the reliability is also the sample weight downstream.
- `evaluate.py` scores everything out-of-fold with R² measured against the training-fold mean. `SparseIncrement` (baseline ridge + best 3 traits on its residuals, selected inside each fold) is the model the permutation test uses; the null shuffles which player owns which tracking profile.
- `clips.py` + `fingerprint.py`: a two-tower contrastive network matching a player's set of Combine clips to their set of game clips. Inputs are orientation-free channels (speed, tangential acceleration, turn rate, lateral acceleration). Batches and evaluation line-ups are always **same role (edge / interior DL, tackle / interior OL, CB / S, WR, TE) and draft class**, so the model cannot win on position, body type or season. Embeddings from different fold or seed models are in different coordinate systems and must never be averaged or pooled; for downstream use, train one model on all players (`train_split_fold(fold=-1)`, legitimate because no outcome enters training).
- `modal_app.py` fans out the five folds × seeds on GPU and the permutation chunks on CPU. `cli.py` holds the functions both it and the local CLI call.

- `game_features.py` computes the *same* per-clip features on game plays as `kinematics.py` does on drills. `matched.py` pairs each drill family with the game situation it rehearses (e.g. comeback/curl drills ↔ hitch routes) and tests one feature measured identically on both sides, net of 40 time, size and role. This like-for-like design is what found signal; comparing drill traits with loosely related game statistics did not.
- `conventional.py` tests every standard Combine number (and pre-specified interactions) against success measures per position, with outcomes centred within on-field role and a min-p permutation test over each position's whole family of tests. `interactions.py` widens the interaction search to every pair of stopwatch numbers, role and drill traits. `career.py` is the pooled 510-player playing-time test. `value.py` is a single pre-specified test (its docstring is the registration; don't edit the spec after the fact), `usage.py` asks whether traits predict deployment instead of results, and `development.py` tests season-over-season change with the first season held fixed.

Lessons that cost time to learn:

- Body-type measurements proxy for role (long arms → tackle → more pressure faced). Centre outcomes within role before correlating.
- Game-side capability should be an upper quantile over plays (0.75–0.95), not the median: most plays don't demand the trait.
- `snaps_per_season` is logged, so its spread is four times wider among late picks than early ones. Any "X matters more for late picks" interaction on it is a scale artefact until it also shows on raw snaps or within pick tiers.
- Contrastive line-ups leak through anything that differs systematically between a player's own pair and other pairs: role, class, which drills are in each half, and how many clips a player has. `fingerprint.py` guards each of these; keep the guards when changing it.

With ~65–80 players per study, adding parameters is costly: a ridge over all traits scores worse than the baseline. Keep trait sets small and any selection inside the cross-validation fold.

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
- **In `combine_tracking.csv`, `a` is an unsigned magnitude and `dis` is not per-frame distance.** Derive signed acceleration from `s` and path length from `x`/`y`. Clips start mid-first-step, not at rest.
- **`dropback_duration` is empty for defenders and for almost all WR rows** (populated for ~59% of OL rows), so separation cannot be adjusted for time to throw.
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
