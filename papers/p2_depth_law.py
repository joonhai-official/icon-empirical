# papers/p2_depth_law.py
#
# Analysis script for paper sections §7 (depth classification, d_c) and §11 (layerwise).
#
# Paper mapping
# -------------
#   §7  Phase Transitions  — depth classification, critical depth d_c per architecture
#   §11 Layer-wise Patterns — per-layer kappa and η_t trajectories
#
#
# Core claim
# ----------
# Depth has two distinct effects depending on architecture:
#
#   kappa_input(d): I(X; Z_d) / d_z  — how much of the original input survives
#   kappa_task(d) : I(Y; Z_d) / d_z  — how much task-relevant info is present
#
# As depth increases:
#   depth_invariant archs (MLP, CNN, GRU, Parallel, Pre-LN): |Δkappa_input| < 5%
#   depth_sensitive archs (Serial, Post-LN): kappa_input drops significantly
#
# The fixed point d* is where the transformer's kappa_input stops decreasing.
# We fit an exponential decay: kappa(d) = kappa_star + A * exp(-d / xi)
# and identify d* as roughly 3*xi (90% of the way to the asymptote).
#
# d_effective
# -----------
#   d_eff(l) = I(X; h_0) - I(X; h_l)   [nats]
# Derived from layerwise kappa measurements.  Always >= 0 by DPI.
# The fixed point is the layer where delta_d_eff < D_EFF_EPS.
#
# Usage
#   python papers/p2_depth_law.py --data results/full.jsonl [--out results/analysis_p2.json]

import argparse
import os
import sys
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.config import SIGMA_SWEEP_WIDTH, ALPHA_R2_MIN
except ImportError:
    SIGMA_SWEEP_WIDTH = 256
    ALPHA_R2_MIN = 0.99


# ---------------------------------------------------------------------------
# Exponential decay fitting: kappa(d) = kappa_star + A * exp(-d / xi)
# ---------------------------------------------------------------------------

def exp_decay_fit(depths: List[int], kappas: List[float]) -> Dict:
    """
    Fit kappa(d) = kappa_star + A * exp(-d / xi) by grid search over
    kappa_star and xi, solving for A analytically at each grid point.

    kappa_star is the asymptotic value (fixed point kappa).
    A is the initial drop amplitude.
    xi is the decay depth — units are "layers".
    d* = 3*xi is a practical definition of when the transient is over.
    """
    if len(depths) < 4:
        return {"kappa_star": kappas[-1], "A": 0.0, "xi": 1.0,
                "r2": 0.0, "d_star": None}

    best: Dict = {"r2": -999.0}

    # candidate asymptotes: last few values and the overall min
    candidates = [kappas[-1], kappas[-2],
                  min(kappas), sum(kappas[-3:]) / 3]

    for kstar in candidates:
        for xi in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0]:
            # A = argmin_A  sum( (kappa_i - kstar - A*exp(-d_i/xi))^2 )
            #   = sum( exp(-d_i/xi) * (kappa_i - kstar) )
            #   / sum( exp(-d_i/xi)^2 )
            xs    = [math.exp(-d / xi) for d in depths]
            ys    = [k - kstar         for k in kappas]
            A_num = sum(x * y for x, y in zip(xs, ys))
            A_den = sum(x * x for x in xs)
            if A_den < 1e-12:
                continue
            A = A_num / A_den

            preds  = [kstar + A * x for x in xs]
            mean_k = sum(kappas) / len(kappas)
            ss_res = sum((k - p) ** 2 for k, p in zip(kappas, preds))
            ss_tot = sum((k - mean_k) ** 2 for k in kappas)
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            if r2 > best["r2"]:
                best = {
                    "kappa_star": round(kstar, 6),
                    "A":          round(A,     6),
                    "xi":         round(xi,    4),
                    "r2":         round(r2,    6),
                    "d_star":     round(3 * xi, 1),  # practical fixed-point definition
                }

    return best


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Depth Law analysis: kappa vs depth profiles, exp decay fit, d_effective.

    Returns depth_sensitive/depth_invariant arch classification and d* estimates.
    """
    # reference slice: w=256, relu, cifar10 — isolates depth effect cleanly
    # Saturated measurements plateau at log(N)/d_z and do not reflect true
    # representation capacity — they would make all depths look equivalent.
    ref = [r for r in records
           if r.get("exp_type") == "main"
           and r.get("width")      == SIGMA_SWEEP_WIDTH
           and r.get("activation") == "relu"
           and r.get("dataset")    == "cifar10"
           and r.get("kappa_input") is not None
           and r.get("sanity_passed")]

    out = {}

    # 1. kappa_input(d) and kappa_task(d) per arch ----------------------------
    print("\n=== 1. kappa_input and kappa_task vs depth ===")
    by_arch: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for r in ref:
        by_arch[r["arch"]][r["depth"]].append(r)

    profiles: Dict[str, Dict] = {}
    for arch in sorted(by_arch):
        depths   = sorted(by_arch[arch])
        ki_means = []
        kt_means = []
        for d in depths:
            rs  = by_arch[arch][d]
            ki  = [r["kappa_input"] for r in rs if r.get("kappa_input")]
            kt  = [r["kappa_task"]  for r in rs if r.get("kappa_task")]
            ki_means.append(sum(ki) / len(ki) if ki else 0.0)
            kt_means.append(sum(kt) / len(kt) if kt else 0.0)

        ki_change_pct = ((ki_means[-1] - ki_means[0]) / ki_means[0] * 100
                         if ki_means[0] > 0 else 0.0)
        kt_change_pct = ((kt_means[-1] - kt_means[0]) / kt_means[0] * 100
                         if kt_means[0] > 0 else 0.0)

        print(f"\n  {arch}  (Δkappa_input={ki_change_pct:+.1f}%  "
              f"Δkappa_task={kt_change_pct:+.1f}%)")
        print(f"  {'d':>4}  {'ki':>9}  {'kt':>9}  {'kt/ki':>7}")
        for i, d in enumerate(depths):
            ki = ki_means[i]
            kt = kt_means[i]
            ratio = kt / ki if ki > 0 else 0.0
            print(f"  {d:>4}  {ki:>9.5f}  {kt:>9.5f}  {ratio:>7.3f}")

        profiles[arch] = {
            "depths":         depths,
            "kappa_input":    [round(k, 6) for k in ki_means],
            "kappa_task":     [round(k, 6) for k in kt_means],
            "ki_change_pct":  round(ki_change_pct, 2),
            "kt_change_pct":  round(kt_change_pct, 2),
        }

    out["profiles"] = profiles

    # 2. Arch classification ---------------------------------------------------
    print("\n=== 2. Arch classification ===")
    invariant, sensitive = [], []
    for arch, p in profiles.items():
        if abs(p["ki_change_pct"]) < 5.0:
            invariant.append(arch)
            label = "depth_invariant"
        else:
            sensitive.append(arch)
            label = "depth_sensitive"
        print(f"  {arch:25s}  {label}  (Δki={p['ki_change_pct']:+.1f}%)")

    out["depth_invariant"] = invariant
    out["depth_sensitive"]  = sensitive

    # 3. Exponential decay fit for sensitive archs ----------------------------
    print("\n=== 3. Exponential decay fit: kappa(d) = kappa* + A*exp(-d/xi) ===")
    exp_fits: Dict[str, Dict] = {}
    for arch in sensitive:
        p = profiles.get(arch, {})
        if not p:
            continue
        fit = exp_decay_fit(p["depths"], p["kappa_input"])
        exp_fits[arch] = fit
        print(f"  {arch:25s}  kappa*={fit['kappa_star']:.5f}  "
              f"A={fit['A']:.5f}  xi={fit['xi']:.2f}  "
              f"R2={fit['r2']:.4f}  d*~{fit['d_star']}")

    out["exp_fits"] = exp_fits

    # 4. d_effective from layerwise data --------------------------------------
    print("\n=== 4. d_effective (from layerwise kappa) ===")
    lw_recs = [r for r in ref
               if r.get("d_effective") and len(r["d_effective"]) > 0
               and r.get("seed") == 0]

    d_eff_by_arch: Dict[str, List] = defaultdict(list)
    for r in lw_recs:
        d_eff_by_arch[r["arch"]].append((r["depth"], r["d_effective"]))

    d_eff_results: Dict[str, Dict] = {}
    for arch, items in sorted(d_eff_by_arch.items()):
        print(f"\n  {arch}:")
        # use depth=8 as the representative (transformer typically plateaus there)
        target = next((de for d, de in sorted(items) if d == 8), None)
        if target is None and items:
            target = sorted(items)[-1][1]
        if not target:
            continue
        d_star_tap = None
        for entry in target:
            is_fp = entry.get("is_fixed_point", False)
            if is_fp and d_star_tap is None:
                d_star_tap = entry["tap_name"]
            print(f"    {entry['tap_name']:15s}  d_eff={entry['d_effective']:.4f}  "
                  f"delta={entry['delta_d_eff']:.4f}"
                  + ("  <- d*" if is_fp and entry['tap_name'] == d_star_tap else ""))
        d_eff_results[arch] = {"d_star_tap": d_star_tap, "profile": target}

    out["d_effective"] = d_eff_results

    return out


def main() -> None:
    """Load full.jsonl, run analyze(), save results/analysis_p2.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/full.jsonl")
    p.add_argument("--out",  default="results/analysis_p2.json")
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
