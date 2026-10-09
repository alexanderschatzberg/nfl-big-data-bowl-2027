import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed

from . import clips, combine, evaluate, outcomes, screen
from .data import build_parquet
from .hypotheses import HYPOTHESES

PERM_REPEATS = 3  # CV repeats per permutation; the observed statistic uses the same
PERM_CHUNK = 10


def prep() -> None:
    build_parquet()
    combine.build()
    outcomes.build()
    clips.build()


def _blocks(name: str, outcome: str):
    p = evaluate.prepare(screen.gated_design(HYPOTHESES[name]), outcome)
    return p["blocks"]["stopwatch+draft"], p["blocks"]["tracking"], p["y"], p["w"]


def fit_outcome(name: str, outcome: str) -> dict:
    hyp = HYPOTHESES[name]
    res = evaluate.fit(screen.gated_design(hyp), outcome)
    stage = "transfer" if outcome == hyp.transfer else "value"
    return res | {"hypothesis": name, "title": hyp.title, "stage": stage,
                  "observed": evaluate.incremental_r2(*_blocks(name, outcome), PERM_REPEATS)}


def perm_chunk(name: str, outcome: str, seeds: list[int]) -> list[float]:
    return evaluate.permuted_increments(*_blocks(name, outcome), seeds, PERM_REPEATS)


def perm_chunks(n_perm: int) -> list[list[int]]:
    return [list(range(i, min(i + PERM_CHUNK, n_perm))) for i in range(0, n_perm, PERM_CHUNK)]


def summary_line(r: dict) -> str:
    perm = r.get("permutation", {})
    p = f"  perm p={perm['p_value']:.3f}" if perm.get("p_value") is not None else ""
    return (f"{r['hypothesis']:<10} {r['stage']:<8} {r['outcome']:<22} n={r['n_players']:<3} "
            f"R2 base={r['r2']['stopwatch+draft']:+.3f} sparse={r['r2']['sparse']:+.3f} "
            f"dR2={r['delta_r2']['sparse']:+.3f}{p}  {r['selected']}")


def run_fit(names: list[str], n_perm: int, n_jobs: int, out: Path) -> list[dict]:
    results = []
    for name in names:
        for outcome in HYPOTHESES[name].outcomes:
            res = fit_outcome(name, outcome)
            if n_perm:
                null = Parallel(n_jobs=n_jobs)(delayed(perm_chunk)(name, outcome, c) for c in perm_chunks(n_perm))
                evaluate.add_permutation(res, res["observed"], [d for c in null for d in c])
            results.append(res)
            print(summary_line(res))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def run_screen(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, hyp in HYPOTHESES.items():
        t = screen.screen(hyp)
        t.to_csv(out / f"screen_{name}.csv", index=False)
        print(f"\n== {hyp.title}")
        print(t.drop(columns=[c for c in t if c.startswith("p__")] + ["n_icc"]).round(2).to_string(index=False))


def run_fingerprint(folds: list[int], steps: int | None, out: Path) -> None:
    from . import fingerprint  # torch is an optional dependency

    overrides = {"steps": steps} if steps else {}
    summary = fingerprint.save([fingerprint.train_fold(f, **overrides) for f in folds])
    out.mkdir(parents=True, exist_ok=True)
    (out / "fingerprint.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(prog="bdb27")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep", help="CSV -> Parquet, Combine features, NFL outcomes, clip tensors")
    sub.add_parser("screen", help="cheap trait screens: repeatability, redundancy, transfer, value")
    n = sub.add_parser("fingerprint", help="train the contrastive movement-fingerprint network")
    n.add_argument("--folds", nargs="*", type=int, default=list(range(5)))
    n.add_argument("--steps", type=int)
    f = sub.add_parser("fit", help="test transfer and value on traits that pass the screen")
    f.add_argument("--hyp", nargs="*", default=list(HYPOTHESES), choices=list(HYPOTHESES))
    f.add_argument("--n-perm", type=int, default=200)
    f.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep()
    elif args.cmd == "screen":
        run_screen(args.out)
    elif args.cmd == "fingerprint":
        run_fingerprint(args.folds, args.steps, args.out)
    else:
        run_fit(args.hyp, args.n_perm, args.n_jobs, args.out)
