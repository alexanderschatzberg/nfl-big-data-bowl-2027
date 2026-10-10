# Kaggle Writeup form: what to enter

| Field | Value |
|---|---|
| **Title** (≤ 80) | Brakes Travel, Separation Doesn't Follow |
| **Subtitle** (≤ 140) | Combine tracking predicts how NFL receivers brake almost as well as a season of game data. Braking still doesn't buy separation. |
| **Card and thumbnail image** | `report/figures/card_560x280.png` (560 × 280) |
| **Submission track** | University track |
| **Project description** | Paste `report/writeup.md` |
| **DOI citation** | Your choice; not needed for judging |

## Media gallery

Upload in this order (the writeup refers to them as Figures 1–5):

1. `report/figures/fig2_braking_transfer.png` (lead image: the headline result)
2. `report/figures/fig3_benchmark.png`
3. `report/figures/fig5_chain.png`
4. `report/figures/fig1_stopwatch.png`
5. `report/figures/fig4_robustness.png`

## Images inside the description

`writeup.md` embeds the five figures with relative paths (`figures/fig1_stopwatch.png` …). Kaggle's editor will not resolve those. When pasting, either drag each PNG into the editor at its place, or replace each path with the image's public URL. The count is 5 figures + 1 table (limit: fewer than 10).

## Attachments

| Attachment | What | Status |
|---|---|---|
| **Public Kaggle Notebook** (required) | Upload `report/notebook.ipynb` as a Kaggle Notebook on the competition, make it public, link it under Project links | Not yet created. Runs locally end to end; **not tested on Kaggle** |
| **GitHub repository** | https://github.com/alexanderschatzberg/nfl-big-data-bowl-2027 | Public; the work is on `main` |

## Before the notebook will run on Kaggle

1. The first cell installs the package with `pip install git+https://github.com/alexanderschatzberg/nfl-big-data-bowl-2027@main`, which is pushed.
2. In the Kaggle notebook: **add the competition data** as an input and turn **Internet on** (needed for the pip install). If the competition forbids Internet in notebooks, attach the repository as a Kaggle Dataset instead; the first cell looks for `/kaggle/input/*/src/bdb27` and uses it without installing.
3. Run all. Parquet files and artifacts are written to `/kaggle/working/bdb27` (about 0.5 GB), figures to `/kaggle/working/figures`.

Untested on Kaggle: the install step, whether the pre-installed duckdb/pandas/scikit-learn versions satisfy the package's minimums (pip will upgrade them if not, which may need a kernel restart), memory during the CSV-to-Parquet conversion of the 1 GB game-tracking file, and run time on Kaggle's CPUs (about 10 minutes locally on 8 cores).

## Check against the rules before submitting

- Whether sharing code publicly on GitHub during the competition is allowed (some Kaggle competitions restrict private sharing outside teams; public sharing is normally fine).
- Word count and figure count are in the limits (see the final report from this build).
