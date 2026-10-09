import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed

from . import combine, evaluate, outcomes, screen
from .data import build_parquet
from .hypotheses import HYPOTHESES, load_design

PERM_REPEATS = 3  # CV repeats per permutation; the observed statistic uses the same


def run_fit(names: list[str], n_perm: int, n_jobs: int, out: Path) -> list[dict]:
    results = []
    for name in names:
        hyp = HYPOTHESES[name]
        design = load_design(hyp)
        for outcome in hyp.outcomes:
            res = evaluate.fit(design, outcome) | {"hypothesis": name, "title": hyp.title}
            if n_perm:
                p = evaluate.prepare(design, outcome)
                base, track = p["blocks"]["stopwatch+draft"], p["blocks"]["tracking"]
                observed = evaluate.incremental_r2(base, track, p["y"], p["w"], PERM_REPEATS)
                chunks = [list(range(i, min(i + 10, n_perm))) for i in range(0, n_perm, 10)]
                null = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate.permuted_increments)(base, track, p["y"], p["w"], c, PERM_REPEATS) for c in chunks)
                evaluate.add_permutation(res, observed, [d for c in null for d in c])
            results.append(res)
            print(summary_line(res))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def summary_line(r: dict) -> str:
    perm = r.get("permutation", {})
    p = f"  perm p={perm['p_value']:.3f}" if perm.get("p_value") is not None else ""
    return (f"{r['hypothesis']:<10} {r['outcome']:<22} n={r['n_players']:<3} rel={r['mean_reliability']:.2f}  "
            f"R2 base={r['r2']['stopwatch+draft']:+.3f} dense={r['r2']['dense']:+.3f} "
            f"sparse={r['r2']['sparse']:+.3f}  dR2={r['delta_r2']['sparse']:+.3f}{p}  {r['selected']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="bdb27")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep", help="CSV -> Parquet, Combine features, NFL outcomes")
    s = sub.add_parser("screen", help="cheap trait screens: repeatability, redundancy, transfer, value")
    s.add_argument("--out", type=Path, default=Path("outputs"))
    f = sub.add_parser("fit", help="evaluate hypotheses")
    f.add_argument("--hyp", nargs="*", default=list(HYPOTHESES), choices=list(HYPOTHESES))
    f.add_argument("--n-perm", type=int, default=200)
    f.add_argument("--n-jobs", type=int, default=-1)
    f.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    if args.cmd == "prep":
        build_parquet()
        combine.build()
        outcomes.build()
    elif args.cmd == "screen":
        args.out.mkdir(parents=True, exist_ok=True)
        for name, hyp in HYPOTHESES.items():
            t = screen.screen(hyp)
            t.to_csv(args.out / f"screen_{name}.csv", index=False)
            print(f"\n== {hyp.title}")
            print(t.drop(columns=[c for c in t if c.startswith("p__")] + ["n_icc"]).round(2).to_string(index=False))
    else:
        run_fit(args.hyp, args.n_perm, args.n_jobs, args.out)
