# experiments/aggregate.py
#
# Merge per-shard JSONL files into a single full_results.jsonl, then
# print a quick sanity report covering record counts, error rates,
# and a Width Law spot-check on the merged data.
#
# Usage
#   python experiments/aggregate.py [--in_dir results/] [--out results/full.jsonl]
#
# The Width Law check computes kappa * width for every (arch, depth=4,
# activation=relu, dataset=cifar10, seed=0) record and reports the
# coefficient of variation.  CV < 0.05 confirms the law holds.

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.config import SIGMA_SWEEP_WIDTH, SIGMA_SWEEP_DEPTH
except ImportError:
    SIGMA_SWEEP_WIDTH = 256
    SIGMA_SWEEP_DEPTH = 4


def load_shards(in_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Read all shard_*.jsonl files from in_dir.

    Returns (records, errors).  Records have kappa_input; errors do not.
    Reads only shard_* files to avoid re-ingesting a prior full.jsonl.
    """
    records, errors = [], []
    # Only read shard files — exclude any previously aggregated full.jsonl.
    for fname in sorted(f for f in os.listdir(in_dir)
                        if f.endswith(".jsonl") and f.startswith("shard_")):
        with open(os.path.join(in_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "error" in r and "kappa_input" not in r:
                        errors.append(r)
                    else:
                        records.append(r)
                except Exception:
                    pass
    return records, errors


def report(records: List[Dict], errors: List[Dict]) -> None:
    """Print aggregate statistics and a Width Law spot-check.

    Width Law: CV = (max−min)/mean of κ·w should be < 0.05.
    CV >= 0.05 signals a data or measurement problem.
    """
    total = len(records) + len(errors)
    print(f"\nRecords  : {len(records):,}")
    print(f"Errors   : {len(errors):,}  ({100*len(errors)/max(total,1):.1f}%)")

    sanity_ok = sum(1 for r in records if r.get("sanity_passed") is True)
    saturated = sum(1 for r in records if r.get("saturated")     is True)
    print(f"Sanity   : {sanity_ok}/{len(records)} passed")
    print(f"Saturated: {saturated}/{len(records)}")

    print("\n--- exp_type ---")
    for k, v in sorted(Counter(r.get("exp_type") for r in records).items()):
        print(f"  {k}: {v:,}")

    print("\n--- dataset ---")
    for k, v in sorted(Counter(r.get("dataset") for r in records).items()):
        print(f"  {k}: {v:,}")

    print("\n--- arch ---")
    for k, v in sorted(Counter(r.get("arch") for r in records).items()):
        print(f"  {k}: {v:,}")

    # Width Law spot-check
    print("\n--- Width Law spot-check (depth=4, relu, cifar10, seed=0) ---")
    target = [
        r for r in records
        if (r.get("exp_type")   == "main"
            and r.get("dataset")    == "cifar10"
            and r.get("activation") == "relu"
            and r.get("depth")      == SIGMA_SWEEP_DEPTH
            and r.get("seed")       == 0
            and r.get("kappa_input") is not None
            and r["kappa_input"] > 0
            and not r.get("saturated", False))
    ]
    by_arch: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in target:
        by_arch[r["arch"]][r["width"]] = r["kappa_input"]

    for arch, wk in sorted(by_arch.items()):
        prods = sorted([(w, k, w * k) for w, k in wk.items()])
        vals  = [p[2] for p in prods]
        if len(vals) >= 2:
            mean_c = sum(vals) / len(vals)
            cv     = (max(vals) - min(vals)) / mean_c
            flag   = "OK" if cv < 0.05 else "WARN"
            print(f"  [{flag}] {arch:25s}  C~={mean_c:.3f}  CV={cv:.4f}")
            for w, k, prod in prods:
                print(f"          w={w:4d}  kappa={k:.5f}  k*w={prod:.3f}")


def main() -> None:
    """Merge shard JSONL files into full.jsonl and print a sanity report.

    For the main experiment (P1-P5):
        python experiments/aggregate.py --in_dir results/ --out results/full.jsonl

    For the P6 thermodynamics experiment (run separately):
        python experiments/aggregate.py --in_dir results_p6/ --out results_p6/full.jsonl
    """
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="results/",
                   help="Directory containing shard_*.jsonl files")
    p.add_argument("--out",    default="results/full.jsonl",
                   help="Output path for merged full.jsonl")
    args = p.parse_args()

    print(f"Loading shards from {args.in_dir} ...")
    records, errors = load_shards(args.in_dir)

    report(records, errors)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(records):,} records -> {args.out}")

    if errors:
        err_path = args.out.replace(".jsonl", "_errors.jsonl")
        with open(err_path, "w") as f:
            for r in errors:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(errors):,} errors  -> {err_path}")


if __name__ == "__main__":
    main()
