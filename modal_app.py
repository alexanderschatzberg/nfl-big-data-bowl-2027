"""Run the full experiment on Modal.

    modal volume create bdb27-data
    modal volume put bdb27-data data/parquet /parquet     # after `bdb27 prep` locally, or upload the CSVs
    modal run modal_app.py                                 # everything
    modal run modal_app.py --n-perm 5000 --seeds 5         # the large run

Raw CSVs can be uploaded to the volume root instead; `prep` converts whatever is missing.
GPU functions need a payment method on the Modal account; `BDB27_GPU= modal run ...` trains on CPU.
"""
import json
import os

import modal

app = modal.App("bdb27")
volume = modal.Volume.from_name("bdb27-data", create_if_missing=True)
VOL = "/vol"

base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("duckdb>=1.1", "joblib>=1.4", "numpy>=2.0", "pandas>=2.2", "pyarrow>=17",
                 "scikit-learn>=1.5", "scipy>=1.13")
    .env({"BDB27_DATA": VOL})
)
cpu_image = base.add_local_python_source("bdb27")
gpu_image = base.pip_install("torch>=2.4").add_local_python_source("bdb27")
common = dict(volumes={VOL: volume}, timeout=3600)
GPU = os.environ.get("BDB27_GPU", "T4") or None
train_resources = dict(gpu=GPU) if GPU else dict(cpu=8, memory=16384)


@app.function(image=cpu_image, cpu=8, memory=32768, **common)
def prep():
    from bdb27 import cli

    cli.prep()
    volume.commit()


@app.function(image=gpu_image, **train_resources, **common)
def train_fold(fold: int, seed: int, overrides: dict) -> dict:
    from bdb27 import fingerprint

    return fingerprint.train_fold(fold, seed, **overrides)


@app.function(image=gpu_image, **common)
def save_fingerprint(results: list[dict]) -> dict:
    from bdb27 import fingerprint

    summary = fingerprint.save(results)
    volume.commit()
    return summary


@app.function(image=cpu_image, cpu=2, **common)
def fit_outcome(name: str, outcome: str) -> dict:
    from bdb27 import cli

    volume.reload()
    return cli.fit_outcome(name, outcome)


@app.function(image=cpu_image, cpu=2, **common)
def perm_chunk(name: str, outcome: str, seeds: list[int]) -> list[float]:
    from bdb27 import cli

    return cli.perm_chunk(name, outcome, seeds)


@app.local_entrypoint()
def main(n_perm: int = 2000, seeds: int = 3, steps: int = 0, skip_prep: bool = False):
    from bdb27 import cli, evaluate
    from bdb27.hypotheses import HYPOTHESES

    if not skip_prep:
        prep.remote()

    # Stage 2, learned: can held-out players be re-identified from Combine movement?
    overrides = {"steps": steps} if steps else {}
    jobs = [(fold, seed, overrides) for seed in range(seeds) for fold in range(5)]
    fingerprint_summary = save_fingerprint.remote(list(train_fold.starmap(jobs)))
    print(json.dumps(fingerprint_summary, indent=2))

    # Stages 2 and 3, per trait: transfer and value, each with its own permutation null.
    pairs = [(name, outcome) for name, hyp in HYPOTHESES.items() for outcome in hyp.outcomes]
    results = list(fit_outcome.starmap(pairs))
    chunks = cli.perm_chunks(n_perm)
    for res, (name, outcome) in zip(results, pairs):
        null = [d for c in perm_chunk.starmap([(name, outcome, c) for c in chunks]) for d in c]
        evaluate.add_permutation(res, res["observed"], null)
        print(cli.summary_line(res))

    with open("outputs/modal_results.json", "w") as f:
        json.dump({"fingerprint": fingerprint_summary, "results": results}, f, indent=2, default=float)
