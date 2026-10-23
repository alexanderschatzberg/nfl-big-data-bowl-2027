# Predicting How Receivers Move: Combine Drills Against a Season of NFL Tracking

Code for an NFL Big Data Bowl 2027 submission. The writeup is [`report/writeup.md`](report/writeup.md), and [`report/notebook.ipynb`](report/notebook.ipynb) reproduces every number and figure in it.

## The finding

I measure one movement trait, peak sustained deceleration over 0.5 s, the same way in the Combine comeback and curl drills and on regular-season hitch routes. For 54 wide receivers from the 2023–2025 draft classes:

- Drill braking predicts game braking: Spearman ρ = 0.42, and partial ρ = 0.43 controlling for 40 time, height, weight and alignment (95% bootstrap CI 0.18–0.63, permutation p = 0.005). It is one of two pairings out of 29 that survive Benjamini–Hochberg correction.
- On the same seasons, the drills predict next-season braking at ρ = 0.33, against 0.56 for the previous season. Controlling for that season, the drills add nothing.
- Braking does not predict separation. That includes a man-coverage test that was written and committed before it was run.

## Where to look

| What | Where |
|---|---|
| Submission text | `report/writeup.md` |
| Every quoted number, as computed | `report/numbers.json` (written by `report.py`) |
| Figures | `report/figures/`, drawn by `src/bdb27/figures.py` and `traces.py` |
| Headline drill-to-game comparison, robustness, season benchmark | `src/bdb27/matched.py` |
| Per-frame movement features (drill and game) | `src/bdb27/kinematics.py`, `game_features.py` |
| Drill aggregation, z-scoring within drill × position × class, cut-off clip filter | `src/bdb27/combine.py` |
| Pre-specified man-coverage test | `src/bdb27/value.py`. The docstring is the specification, committed in `555f108` before the test was run in `eff853d` |
| Stopwatch numbers against outcomes, with family-wise permutation p | `src/bdb27/conventional.py` |
| Wider searches reported as null (interactions, usage, development, neural model) | `interactions.py`, `usage.py`, `development.py`, `fingerprint.py` |

The writeup's "Everything else I tested" table lists every test family in the notebook, 5,826 tests in all.

## Reproducing

Needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). Put the competition CSVs in `./data`, or point `BDB27_DATA` at another directory (`BDB27_RAW` if the CSVs are read-only).

```bash
uv sync
uv run bdb27 prep        # CSV -> Parquet, drill and game features (~1 min)
uv run bdb27 matched     # drill-to-game table, 29 pairings
uv run bdb27 robustness  # variants, placebos, season benchmark
uv run bdb27 value       # the pre-specified test
uv run bdb27 figures     # writes report/figures
uv run pytest            # unit tests for the kinematics
```

The full notebook runs in about 10 minutes on 8 cores:

```bash
uv run python -m ipykernel install --sys-prefix --name bdb27
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=bdb27 \
  --ExecutePreprocessor.timeout=3600 report/notebook.ipynb --output notebook.ipynb
```

Results are deterministic: rows are sorted before modelling and all random draws are seeded. One exception is the family-wise permutation p-values for the stopwatch tests, which vary in the third decimal between runs.

## Choices worth knowing

- **Same measurement on both sides.** Comparing drill traits with loosely related game statistics found nothing. Measuring one feature identically in a drill and in the game situation it rehearses is what worked.
- **Game capability is an upper quantile.** Each receiver's 90th-percentile hitch route is used, because most routes don't call for a full stop. The median gives ρ = 0.18 (p = 0.19). `matched.robustness` reports every quantile.
- **Cut-off clips are dropped.** Combine clips shorter than 65% of the drill's median length end before the receiver stops. Removing them (29 of 6,307) lowers the headline from 0.49 to 0.43.
- **Game tracking covers only drafted cohort players.** There is no ball or opponent row, so separation and cushion come from `player_play.csv`.
- **There is no holdout class.** All three draft classes were used during exploration. The 2026 class is the real test.
