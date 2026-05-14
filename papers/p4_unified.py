# papers/p4_unified.py
#
# Analysis script for paper section §6 (Two-Variable Scaling).
#
# Paper mapping
# -------------
#   §6 Two-Variable Scaling kappa(w, d)
#      kappa = C * w^alpha * d^beta   (two-variable OLS in log space)
#
#
# Model
# -----
#   kappa(w, d, arch) = C(arch) * w^alpha * d^beta
#
# in log space:
#   log(kappa) = log(C_arch) + alpha*log(w) + beta*log(d)
#
# We fix C(arch) from the Width Law analysis (§4, papers/p1_width_law.py), then solve for
# alpha and beta jointly via OLS on the residuals:
#   log(kappa) - log(C_arch) = alpha*log(w) + beta*log(d)
#
# This 2-variable OLS uses the normal equations directly (no external solver).
#
# Target: unified R-squared > 0.95.
# Below that threshold, the unified law does not support the hardware design
# claims, so we report honestly and identify which archs drive the residual.
#
# We run the same fit separately for kappa_input and kappa_task.
#
# Usage
#   python papers/p4_unified.py --data results/full.jsonl [--out results/analysis_p4.json]

import argparse
import os
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# C_arch estimation
# ---------------------------------------------------------------------------

def estimate_c_arch(records: List[Dict]) -> Dict[str, float]:
    """
    Estimate C_arch as the mean of kappa * width across all conditions for
    each architecture, restricted to depth=4 and activation=relu so only
    width varies.
    """
    by_arch: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if (r.get("exp_type") == "main"
                and r.get("depth") == 4
                and r.get("activation") == "relu"
                and r.get("kappa_input") is not None
                and r["kappa_input"] > 0
                and r.get("sanity_passed")):
            by_arch[r["arch"]].append(r["width"] * r["kappa_input"])
    return {arch: round(sum(vs) / len(vs), 6) for arch, vs in by_arch.items()}


# ---------------------------------------------------------------------------
# 2-variable OLS in log space
# ---------------------------------------------------------------------------

def ols_alpha_beta(
    log_w:  List[float],
    log_d:  List[float],
    log_yz: List[float],   # log(kappa) - log(C_arch)
) -> Optional[Tuple[float, float, float]]:
    """
    Solve: log_yz = alpha * log_w + beta * log_d  (no intercept)
    via normal equations.

    Returns (alpha, beta, R2) or None if the system is singular.
    """
    n = len(log_w)
    s_ww  = sum(x ** 2 for x in log_w)
    s_dd  = sum(x ** 2 for x in log_d)
    s_wd  = sum(log_w[i] * log_d[i]  for i in range(n))
    s_wyz = sum(log_w[i] * log_yz[i] for i in range(n))
    s_dyz = sum(log_d[i] * log_yz[i] for i in range(n))

    det = s_ww * s_dd - s_wd ** 2
    if abs(det) < 1e-12:
        return None

    alpha = (s_dd * s_wyz - s_wd * s_dyz) / det
    beta  = (s_ww * s_dyz - s_wd * s_wyz) / det

    preds  = [alpha * log_w[i] + beta * log_d[i] for i in range(n)]
    mean_y = sum(log_yz) / n
    ss_res = sum((log_yz[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((log_yz[i] - mean_y)   ** 2 for i in range(n))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return alpha, beta, r2


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Fit κ(w,d,arch) = C·w^α·d^β via 2-variable no-intercept OLS.

    C_arch pre-estimated from depth=4 relu conditions.
    Separately fits kappa_input and kappa_task.
    """
    # Exclude saturated records — they bias alpha toward 0.
    main = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("kappa_input") is not None
            and r["kappa_input"] > 0
            and r.get("sanity_passed")]

    out: Dict = {}

    # 1. C_arch table ----------------------------------------------------------
    print("\n=== 1. C_arch estimates ===")
    c_arch = estimate_c_arch(records)
    for arch, c in sorted(c_arch.items(), key=lambda x: -x[1]):
        print(f"  {arch:25s}  C={c:.4f}")
    out["c_arch"] = c_arch

    # 2. Unified kappa_input fit -----------------------------------------------
    print("\n=== 2. Unified fit: kappa_input = C * w^alpha * d^beta ===")
    rows_i = [
        (r, math.log(r["kappa_input"]),
         math.log(r["width"]),
         math.log(max(r["depth"], 1)))
        for r in main if r.get("arch") in c_arch
    ]
    if len(rows_i) >= 10:
        log_yz = [lk - math.log(c_arch[r["arch"]]) for r, lk, _, _ in rows_i]
        log_w  = [lw for _, _, lw, _  in rows_i]
        log_d  = [ld for _, _, _,  ld in rows_i]
        fit_i  = ols_alpha_beta(log_w, log_d, log_yz)
        if fit_i:
            alpha, beta, r2 = fit_i
            print(f"  alpha = {alpha:.4f}  (theory: -1.0)")
            print(f"  beta  = {beta:.4f}")
            print(f"  R2    = {r2:.4f}  {'OK' if r2 >= 0.95 else 'below 0.95 threshold'}")
            print(f"  n     = {len(rows_i):,}")
            out["kappa_input"] = {"alpha": round(alpha, 6), "beta": round(beta, 6),
                                  "r2": round(r2, 6), "n": len(rows_i)}

    # 3. Unified kappa_task fit ------------------------------------------------
    print("\n=== 3. Unified fit: kappa_task = C * w^alpha * d^beta ===")
    rows_t = [
        (r, math.log(r["kappa_task"]),
         math.log(r["width"]),
         math.log(max(r["depth"], 1)))
        for r in main
        if r.get("arch") in c_arch
        and r.get("kappa_task") is not None
        and r["kappa_task"] > 0
    ]
    if len(rows_t) >= 10:
        log_yz = [lk - math.log(c_arch[r["arch"]]) for r, lk, _, _ in rows_t]
        log_w  = [lw for _, _, lw, _  in rows_t]
        log_d  = [ld for _, _, _,  ld in rows_t]
        fit_t  = ols_alpha_beta(log_w, log_d, log_yz)
        if fit_t:
            alpha, beta, r2 = fit_t
            print(f"  alpha = {alpha:.4f}")
            print(f"  beta  = {beta:.4f}")
            print(f"  R2    = {r2:.4f}")
            out["kappa_task"] = {"alpha": round(alpha, 6), "beta": round(beta, 6),
                                 "r2": round(r2, 6), "n": len(rows_t)}

    # 4. Per-arch residuals from the global fit (diagnose outliers) -----------
    if "kappa_input" in out:
        print("\n=== 4. Per-arch residuals from unified fit ===")
        alpha = out["kappa_input"]["alpha"]
        beta  = out["kappa_input"]["beta"]
        by_arch: Dict[str, List[float]] = defaultdict(list)
        for r, lk, lw, ld in rows_i:
            pred = math.log(c_arch[r["arch"]]) + alpha * lw + beta * ld
            by_arch[r["arch"]].append(lk - pred)
        for arch in sorted(by_arch):
            res  = by_arch[arch]
            mean = sum(res) / len(res)
            std  = math.sqrt(sum((x - mean) ** 2 for x in res) / len(res))
            print(f"  {arch:25s}  mean_resid={mean:+.4f}  std={std:.4f}")

    return out


def main() -> None:
    """Load full.jsonl, run analyze(), save results/analysis_p4.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/full.jsonl")
    p.add_argument("--out",  default="results/analysis_p4.json")
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
