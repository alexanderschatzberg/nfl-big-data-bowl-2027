# Kaggle Writeup form: what to enter

| Field | Value |
|---|---|
| **Title** (≤ 80) | Predicting How Receivers Move: Combine Drills Against a Season of NFL Tracking |
| **Subtitle** (≤ 140) | The Combine comeback and curl drills predict how hard a receiver brakes on NFL hitch routes, though less well than a season of NFL tracking. |
| **Card and thumbnail image** | `report/figures/card_560x280.png` (560 × 280) |
| **Submission track** | University track |
| **Project description** | Paste `report/writeup_kaggle.md` |
| **DOI citation** | Your choice; not needed for judging |

## Media gallery

Upload in this order (file names are unchanged; the benchmark is the lead claim, so it goes first):

1. `report/figures/fig3_benchmark.png` (lead image: the Combine compared with a season of NFL tracking)
2. `report/figures/fig6_traces.png`
3. `report/figures/fig2_braking_transfer.png`
4. `report/figures/fig5_chain.png`
5. `report/figures/fig1_stopwatch.png`
6. `report/figures/fig4_robustness.png`

## Images inside the description

`writeup.md` embeds the six figures with relative paths, which Kaggle cannot resolve. Paste `writeup_kaggle.md` instead: the same text with each image pointing at its raw GitHub URL, pinned to a commit. If the figures or text change, push, then regenerate it with `sed "s#](figures/#](https://raw.githubusercontent.com/alexanderschatzberg/nfl-big-data-bowl-2027/$(git rev-parse HEAD)/report/figures/#" report/writeup.md > report/writeup_kaggle.md`. The count is 6 figures + 1 table (limit: fewer than 10). The text is 1,975 words including the table and headings, not counting image tags (limit: 2,000).

## Attachments

| Attachment | What | Status |
|---|---|---|
| **Public Kaggle Notebook** (required) | Upload `report/notebook.ipynb` as a Kaggle Notebook on the competition, make it public, link it under Project links | Not yet created. Runs locally end to end; **not tested on Kaggle** |
| **GitHub repository** | https://github.com/alexanderschatzberg/nfl-big-data-bowl-2027 | Public; the work is on `main` |

## Before the notebook will run on Kaggle

1. The first cell installs the package with `pip install git+https://github.com/alexanderschatzberg/nfl-big-data-bowl-2027@main`, so the latest commits on `main` must be pushed before running there.
2. In the Kaggle notebook: **add the competition data** as an input and turn **Internet on** (needed for the pip install). If the competition forbids Internet in notebooks, attach the repository as a Kaggle Dataset instead; the first cell looks for `/kaggle/input/*/src/bdb27` and uses it without installing.
3. Run all. Parquet files and artifacts are written to `/kaggle/working/bdb27` (about 0.5 GB), figures to `/kaggle/working/figures`.

Untested on Kaggle: the install step, whether the pre-installed duckdb/pandas/scikit-learn versions satisfy the package's minimums (pip will upgrade them if not, which may need a kernel restart), memory during the CSV-to-Parquet conversion of the 1 GB game-tracking file, and run time on Kaggle's CPUs (about 10 minutes locally on 8 cores).

## Check against the rules before submitting

- Whether sharing code publicly on GitHub during the competition is allowed (some Kaggle competitions restrict private sharing outside teams; public sharing is normally fine).
- Word count and figure count are in the limits (see the final report from this build).
