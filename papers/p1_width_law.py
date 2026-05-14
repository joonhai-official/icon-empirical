# papers/p1_width_law.py
#
# Analysis script for paper sections §3, §4, §5.
#
# Paper mapping
# -------------
#   §3 Activation Functions: The Negative Starting Point  (activation slice in §3 below)
#   §4 Phenomenological Width Scaling                     (global fit, per-arch fits)
#   §5 Architecture-Specific Constants C_arch             (C_arch table)
#
#
# Core claim
# ----------
#   kappa = C_arch * w^alpha,   alpha ~= -1
#
# We fit this in log-log space (log-kappa vs log-width) using ordinary
# least squares, which is exact for a power law.  R-squared is computed
# on the log-log residuals so it reflects fit quality on the multiplicative
# scale, not the absolute scale (which would over-weight large kappa values).
#
# Analyses
# --------
# 1. Global fit   : pool all arch/act/dataset, fit single alpha
# 2. C_arch table : per (arch, activation), averaged over datasets and seeds
# 3. Activation   : does alpha change between relu / gelu / tanh?
# 4. Dataset      : C_arch consistency check across available datasets
# 5. kappa_task   : does the Width Law hold for kappa_task as well?
#
# Usage
#   python papers/p1_width_law.py --data results/full.jsonl [--out results/analysis_p1.json]

import argparse
import os
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.config import ALPHA_THEORY, ALPHA_R2_MIN
except ImportError:
    ALPHA_THEORY = -1.0
    ALPHA_R2_MIN = 0.99


# ---------------------------------------------------------------------------
# Log-log OLS regression: kappa = C * w^alpha
# ---------------------------------------------------------------------------

def log_log_fit(ws: List[int], kappas: List[float]) -> Dict:
    """
    Fit kappa = C * w^alpha via OLS on log(kappa) = alpha*log(w) + log(C).

    Returns alpha, C (the pre-factor), R-squared, and sample count n.
    R-squared is computed on the log-log residuals.
    Requires at least 2 points.
    """
    assert len(ws) == len(kappas) >= 2, "need at least 2 points"

    lw = [math.log(w) for w in ws]
    lk = [math.log(max(k, 1e-12)) for k in kappas]
    n  = len(lw)

    mlw = sum(lw) / n
    mlk = sum(lk) / n

    ssxy = sum((lw[i] - mlw) * (lk[i] - mlk) for i in range(n))
    ssxx = sum((lw[i] - mlw) ** 2             for i in range(n))

    alpha = ssxy / ssxx if ssxx > 1e-12 else 0.0
    log_C = mlk - alpha * mlw
    C     = math.exp(log_C)

    preds  = [alpha * lw[i] + log_C for i in range(n)]
    ss_res = sum((lk[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((lk[i] - mlk)      ** 2 for i in range(n))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {"alpha": round(alpha, 6), "C": round(C, 6),
            "r2": round(r2, 6), "n": n}


def _mean_kappa_by_width(records: List[Dict], key: str = "kappa_input") -> Dict:
    """Group records by width and return {width: mean_kappa}."""
    by_w: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v and v > 0:
            by_w[r["width"]].append(v)
    return {w: sum(vs) / len(vs) for w, vs in by_w.items()}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Run all Width Law analyses (global fit, C_arch table, activation, dataset, kappa_task).

    Returns a dict of fit results keyed by analysis name.
    """
    # restrict to main experiment, depth=4 (mid-range), to isolate width effect
    # Exclude saturated measurements from Width Law fitting.
    # When MI approaches log(batch_size), kappa plateaus at log(N)/d_z
    # regardless of true representation capacity, which biases alpha toward 0.
    main = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("depth") == 4
            and r.get("kappa_input") is not None
            and r["kappa_input"] > 0
            and r.get("sanity_passed")]

    out = {}

    # 1. Global fit ------------------------------------------------------------
    print("\n=== 1. Global Width Law fit ===")
    wk = _mean_kappa_by_width(main)
    if len(wk) >= 3:
        fit = log_log_fit(sorted(wk), [wk[w] for w in sorted(wk)])
        print(f"  alpha = {fit['alpha']:.4f}  (theory: {ALPHA_THEORY})")
        print(f"  C     = {fit['C']:.4f}")
        print(f"  R2    = {fit['r2']:.6f}  {'OK' if fit['r2'] >= ALPHA_R2_MIN else 'WARN'}")
        out["global"] = fit

    # 2. C_arch table ----------------------------------------------------------
    print("\n=== 2. C_arch table (per arch x activation) ===")
    c_table: Dict[str, Dict] = {}
    for arch in sorted({r["arch"] for r in main}):
        for act in sorted({r["activation"] for r in main}):
            sub = [r for r in main if r["arch"] == arch and r["activation"] == act]
            wk  = _mean_kappa_by_width(sub)
            if len(wk) < 3:
                continue
            fit = log_log_fit(sorted(wk), [wk[w] for w in sorted(wk)])
            key = f"{arch}|{act}"
            c_table[key] = fit
            print(f"  {arch:25s} {act:5s}  C={fit['C']:.4f}  "
                  f"alpha={fit['alpha']:.4f}  R2={fit['r2']:.4f}")
    out["c_arch"] = c_table

    # 3. Activation comparison -------------------------------------------------
    print("\n=== 3. alpha by activation ===")
    act_fits: Dict[str, Dict] = {}
    for act in ["relu", "gelu", "tanh"]:
        sub = [r for r in main if r["activation"] == act]
        wk  = _mean_kappa_by_width(sub)
        if len(wk) >= 3:
            fit = log_log_fit(sorted(wk), [wk[w] for w in sorted(wk)])
            act_fits[act] = fit
            print(f"  {act:5s}  alpha={fit['alpha']:.4f}  C={fit['C']:.4f}  R2={fit['r2']:.4f}")
    out["by_activation"] = act_fits

    # 4. Dataset comparison ----------------------------------------------------
    print("\n=== 4. Width Law by dataset ===")
    ds_fits: Dict[str, Dict] = {}
    for ds in sorted({r["dataset"] for r in main}):
        sub = [r for r in main if r["dataset"] == ds]
        wk  = _mean_kappa_by_width(sub)
        if len(wk) >= 3:
            fit = log_log_fit(sorted(wk), [wk[w] for w in sorted(wk)])
            ds_fits[ds] = fit
            print(f"  {ds:15s}  alpha={fit['alpha']:.4f}  R2={fit['r2']:.4f}")
    out["by_dataset"] = ds_fits

    # 5. kappa_task Width Law --------------------------------------------------
    print("\n=== 5. kappa_task Width Law ===")
    task = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("depth") == 4
            and r.get("kappa_task") is not None
            and r["kappa_task"] > 0
            and r.get("sanity_passed")]
    wk = _mean_kappa_by_width(task, key="kappa_task")
    if len(wk) >= 3:
        fit = log_log_fit(sorted(wk), [wk[w] for w in sorted(wk)])
        out["kappa_task_global"] = fit
        print(f"  alpha={fit['alpha']:.4f}  C={fit['C']:.4f}  R2={fit['r2']:.4f}")

    return out


def main() -> None:
    """Load full.jsonl, run analyze(), save results/analysis_p1.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/full.jsonl")
    p.add_argument("--out",  default="results/analysis_p1.json")
    args = p.parse_args()

    records = []
    with open(args.data) as f:
        for line in f:
            try:    records.append(json.loads(line))
            except: pass
    print(f"Loaded {len(records):,} records")

    result = analyze(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
