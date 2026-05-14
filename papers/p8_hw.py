# papers/p8_hw.py
#
# Analysis script for paper section §10 (Inverse Design).
#
# Paper mapping
# -------------
#   §10 Inverse Design — given kappa_target + memory budget, find (arch, w, d)
#
#
# This paper applies the empirical laws from P1-P7 to practical design problems:
#
# 1. Optimal width under memory constraint
#    Given memory budget M (bytes) and target kappa_target,
#    solve for w* = (kappa_target / C_arch)^{1/alpha}.
#    Verify that w* satisfies the memory constraint.
#
# 2. Architecture selection by C_arch
#    For a fixed memory budget, rank architectures by C_arch.
#    Higher C_arch -> more information per parameter -> better efficiency.
#
# 3. MaxEnt efficiency as hardware utilisation metric
#    eta = kappa / kappa_max ~ 0.85-0.95 across all architectures.
#    Interpretation: hardware utilisation ceiling is ~90% of theoretical max.
#    Designing for eta > 0.95 wastes engineering effort.
#
# 4. Critical depth as pruning boundary (from P6)
#    Layers beyond d_c carry kappa ~ 0 and are candidates for removal.
#    Reports parameter savings from pruning at d_c.
#
# 5. Inverse design: given (kappa_target, memory_budget), output (arch, w, d)
#    Uses the Unified Law kappa = C * w^alpha * d^beta to solve jointly.
#
# Usage
#   python papers/p8_hw.py --data results/full.jsonl \
#                          --data_p6 results_p6/full.jsonl \
#                          [--out results/analysis_p8.json]

import argparse
import os
import sys
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bytes per parameter (float32)
BYTES_PER_PARAM = 4



_ALPHA = -0.9894   # Width Law exponent from P1
_BETA  = -0.0198   # Depth Law exponent from P4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_count(arch: str, width: int, depth: int) -> int:
    """Analytical parameter count for each architecture."""
    w, d = width, depth
    if arch == "mlp":
        return 3 * w * 16 + d * 2 * (64 * 64 + w * 4 * w + 4 * w * w) + w * 10
    elif arch == "cnn":
        n1 = max(1, (d + 2) // 3)
        n2 = max(0, (d - n1 + 1) // 2)
        n3 = max(0, d - n1 - n2)
        return (3 * w * 9
                + n1 * 2 * w * w * 9
                + n2 * 2 * (w * 2) * (w * 2) * 9
                + n3 * 2 * (w * 4) * (w * 4) * 9
                + w * 4 * w + w * 10)
    elif arch == "gru":
        return 48 * w + min(d, 8) * 3 * (w * w + w * w + w) + w * 10
    elif arch in ("transformer_preln", "transformer_postln"):
        return (3 * w * 16 + 64 * w
                + d * (4 * w * w + 2 * (w * 4 * w + 4 * w * w))
                + w * 10)
    elif arch == "serial":
        return 3 * w * 16 + d * (w * 4 * w + 4 * w * w) + w * 10
    elif arch == "parallel":
        return 3 * w * 16 + w * d + d * (w * 4 * w + 4 * w * w) + w * 10
    elif arch == "boltzmann":
        return 3 * 32 * 32 * w + 3 * 32 * 32 + w + w * 10
    return 0


def _memory_mb(arch: str, width: int, depth: int) -> float:
    """Memory in MB for float32 parameters."""
    return _param_count(arch, width, depth) * BYTES_PER_PARAM / (1024 ** 2)


def _optimal_width(c_arch: float, kappa_target: float,
                   alpha: float = _ALPHA) -> int:
    """Solve kappa = C * w^alpha for w.

    w* = (kappa_target / C)^{1/alpha}
    Rounded to nearest power of 2 for hardware alignment.
    """
    if kappa_target <= 0 or c_arch <= 0:
        return 0
    w_exact = (kappa_target / c_arch) ** (1.0 / alpha)
    # Round to nearest power of 2
    log2_w = round(math.log2(w_exact))
    return 2 ** max(4, min(log2_w, 14))   # clamp to [16, 16384]


# ---------------------------------------------------------------------------
# 1. C_arch table and architecture ranking
# ---------------------------------------------------------------------------

def analyze_c_arch(records: List[Dict]) -> Dict:
    """Rank architectures by C_arch — information efficiency constant.

    Higher C_arch means more information per unit width dimension.
    For a fixed memory budget, prefer the architecture with highest C_arch.
    """
    main = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("depth") == 4
            and r.get("activation") == "relu"
            and r.get("kappa_input") is not None
            and r.get("sanity_passed")]

    print(f"\n=== Architecture ranking by C_arch ({len(main)} records) ===")
    results: Dict[str, Dict] = {}

    by_arch_w: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for r in main:
        by_arch_w[r["arch"]][r["width"]].append(r["kappa_input"])

    c_arch_map: Dict[str, float] = {}
    for arch in sorted(by_arch_w):
        ws = sorted(by_arch_w[arch])
        ks = [sum(by_arch_w[arch][w]) / len(by_arch_w[arch][w]) for w in ws]
        # C_arch = mean(kappa * width) — directly from Width Law
        c_vals = [k * w for k, w in zip(ks, ws)]
        c_arch = sum(c_vals) / len(c_vals)
        c_arch_map[arch] = c_arch

    # Rank by C_arch descending
    ranked = sorted(c_arch_map.items(), key=lambda x: -x[1])
    print(f"\n  {'rank':>4}  {'arch':25s}  {'C_arch':>8}  {'relative':>10}")
    best_c = ranked[0][1] if ranked else 1.0
    for i, (arch, c) in enumerate(ranked):
        rel = c / best_c
        print(f"  {i+1:>4}  {arch:25s}  {c:>8.4f}  {rel:>10.3f}")
        results[arch] = {"c_arch": round(c, 4), "rank": i + 1,
                         "relative_efficiency": round(rel, 4)}

    return results


# ---------------------------------------------------------------------------
# 2. Optimal width under memory constraint
# ---------------------------------------------------------------------------

def analyze_memory_design(c_arch_results: Dict[str, Dict]) -> Dict:
    """Compute optimal width w* for common memory budgets.

    Memory budgets: 1MB, 10MB, 100MB, 1GB (typical edge to server).
    For each budget, find w* that maximises kappa under the constraint.
    """
    budgets_mb = [1, 10, 100, 1000]
    depth      = 4   # canonical depth from P1

    print(f"\n=== Optimal width under memory constraint (d={depth}) ===")
    results: Dict[str, Dict] = {}

    for budget_mb in budgets_mb:
        print(f"\n  Budget = {budget_mb} MB:")
        print(f"  {'arch':25s}  {'w*':>6}  {'mem_MB':>8}  {'kappa_est':>10}")
        budget_results = {}
        for arch, info in sorted(c_arch_results.items(),
                                 key=lambda x: -x[1]["c_arch"]):
            c = info["c_arch"]
            # Binary search for largest w that fits in budget
            best_w = 16
            for w in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                if _memory_mb(arch, w, depth) <= budget_mb:
                    best_w = w
            mem    = _memory_mb(arch, best_w, depth)
            kappa  = c * (best_w ** _ALPHA)
            print(f"  {arch:25s}  {best_w:>6d}  {mem:>8.2f}  {kappa:>10.5f}")
            budget_results[arch] = {
                "w_star":    best_w,
                "memory_mb": round(mem,   3),
                "kappa_est": round(kappa, 6),
            }
        results[f"{budget_mb}MB"] = budget_results

    return results


# ---------------------------------------------------------------------------
# 3. MaxEnt efficiency as hardware utilisation metric
# ---------------------------------------------------------------------------

def analyze_hw_efficiency(records: List[Dict]) -> Dict:
    """eta = kappa / kappa_max as hardware utilisation.

    kappa_max = log(512) / d_z  (InfoNCE upper bound).
    eta ~ 0.85-0.95 means hardware achieves 85-95% of theoretical max.
    Designing for eta > 0.95 requires disproportionate engineering effort.
    """
    log_batch = math.log(512)
    main = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("depth") == 4
            and r.get("kappa_input") is not None
            and r.get("d_z") is not None
            and r.get("sanity_passed")]

    print(f"\n=== Hardware utilisation (eta = kappa / kappa_max) ===")
    results: Dict[str, Dict] = {}

    by_arch: Dict[str, List[float]] = defaultdict(list)
    for r in main:
        k_max = log_batch / r["d_z"]
        eta   = r["kappa_input"] / k_max if k_max > 0 else 0.0
        by_arch[r["arch"]].append(eta)

    print(f"\n  {'arch':25s}  {'eta_mean':>9}  {'eta_min':>8}  {'eta_max':>8}")
    for arch in sorted(by_arch):
        etas     = by_arch[arch]
        eta_mean = sum(etas) / len(etas)
        eta_min  = min(etas)
        eta_max  = max(etas)
        print(f"  {arch:25s}  {eta_mean:>9.4f}  {eta_min:>8.4f}  {eta_max:>8.4f}")
        results[arch] = {
            "eta_mean": round(eta_mean, 4),
            "eta_min":  round(eta_min,  4),
            "eta_max":  round(eta_max,  4),
        }

    all_etas = [e for es in by_arch.values() for e in es]
    grand_mean = sum(all_etas) / len(all_etas) if all_etas else 0.0
    print(f"\n  Grand mean eta = {grand_mean:.4f}")
    print(f"  Hardware utilisation ceiling ~ {grand_mean:.0%} of theoretical max.")
    print(f"  Engineering above eta={grand_mean:.2f} yields diminishing returns.")
    results["_summary"] = {"grand_mean_eta": round(grand_mean, 4)}

    return results


# ---------------------------------------------------------------------------
# 4. Pruning boundary from d_c
# ---------------------------------------------------------------------------

def analyze_pruning(records_p6: List[Dict]) -> Dict:
    """Parameter savings from pruning at critical depth d_c.

    Layers 1..d_c carry information (kappa > 0).
    Layers d_c+1..d are dead (kappa ~ 0) — safe to prune.
    Reports parameter count before and after pruning, and savings %.
    """
    phase = [r for r in records_p6
             if r.get("exp_type") == "phase"
             and r.get("kappa_input") is not None]

    print(f"\n=== Pruning boundary analysis ({len(phase)} phase records) ===")
    results: Dict[str, Dict] = {}

    by_aw: Dict[str, Dict[int, Dict[int, List]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in phase:
        by_aw[r["arch"]][r["width"]][r["depth"]].append(r["kappa_input"])

    print(f"\n  {'arch':20s}  {'w':>5}  {'d_full':>7}  {'d_c':>5}  "
          f"{'params_full':>12}  {'params_pruned':>14}  {'saving':>8}")

    for arch in sorted(by_aw):
        for w in sorted(by_aw[arch]):
            by_d    = by_aw[arch][w]
            depths  = sorted(by_d)
            kappas  = [sum(by_d[d]) / len(by_d[d]) for d in depths]
            ref     = kappas[0] if kappas else 0.0
            d_c     = next((d for d, k in zip(depths, kappas)
                            if ref > 0 and k < 0.1 * ref), None)
            if d_c is None:
                continue

            d_full      = max(depths)
            p_full      = _param_count(arch, w, d_full)
            p_pruned    = _param_count(arch, w, d_c - 1)
            saving_pct  = (p_full - p_pruned) / p_full * 100 if p_full > 0 else 0.0

            print(f"  {arch:20s}  {w:>5d}  {d_full:>7d}  {d_c:>5d}  "
                  f"{p_full:>12,d}  {p_pruned:>14,d}  {saving_pct:>7.1f}%")

            results[f"{arch}|w{w}"] = {
                "d_c":          d_c,
                "d_full":       d_full,
                "params_full":  p_full,
                "params_pruned": p_pruned,
                "saving_pct":   round(saving_pct, 2),
            }

    return results


# ---------------------------------------------------------------------------
# 5. Inverse design
# ---------------------------------------------------------------------------

def analyze_inverse_design(c_arch_results: Dict) -> Dict:
    """Given (kappa_target, memory_budget_MB), output optimal (arch, w, d).

    Uses Width Law (P1) and Unified Law (P4):
      kappa = C_arch * w^alpha * d^beta
    Solve for w given kappa_target and d, then verify memory fits budget.
    """
    # Design scenarios
    scenarios = [
        # (name,            kappa_target, budget_mb, n_classes)
        ("Smartphone NPU",   0.02,          50,        10),
        ("Automotive ADAS",  0.015,         200,       10),
        ("Server GPU",       0.005,        4000,       1000),
        ("IoT micro",        0.05,           2,        10),
        ("Space onboard",    0.03,          10,        10),
    ]

    print(f"\n=== Inverse design: (kappa_target, budget) -> (arch, w, d) ===")
    results: Dict[str, Dict] = {}

    c_arch = {arch: info["c_arch"] for arch, info in c_arch_results.items()}
    depths_to_try = [4]   # beta=-0.0198, depth effect negligible

    for name, kappa_t, budget_mb, n_cls in scenarios:
        print(f"\n  Scenario: {name}")
        print(f"    kappa_target={kappa_t}  budget={budget_mb}MB  n_classes={n_cls}")

        best = None
        for arch, C in sorted(c_arch.items(), key=lambda x: -x[1]):
            for d in depths_to_try:
                # Unified Law: kappa = C * w^alpha * d^beta
                # Solve for w: w = (kappa / (C * d^beta))^{1/alpha}
                kappa_w = kappa_t / (C * (d ** _BETA))
                if kappa_w <= 0:
                    continue
                w_exact = kappa_w ** (1.0 / _ALPHA)
                w       = max(16, min(4096, 2 ** round(math.log2(w_exact))))
                mem     = _memory_mb(arch, w, d)
                if mem > budget_mb:
                    continue
                # Serial collapses at d_c=8 (w=512) or d_c=14 (w=256)
                if arch == "serial" and d >= 8:
                    continue
                kappa_achieved = C * (w ** _ALPHA) * (d ** _BETA)
                if best is None or abs(kappa_achieved - kappa_t) < abs(best["kappa_achieved"] - kappa_t):
                    best = {
                        "arch":            arch,
                        "width":           w,
                        "depth":           d,
                        "memory_mb":       round(mem, 2),
                        "kappa_achieved":  round(kappa_achieved, 6),
                        "kappa_target":    kappa_t,
                        "kappa_error_pct": round(abs(kappa_achieved - kappa_t) / kappa_t * 100, 2),
                    }

        if best:
            print(f"    -> arch={best['arch']}  w={best['width']}  d={best['depth']}")
            print(f"       mem={best['memory_mb']}MB  "
                  f"kappa={best['kappa_achieved']} (target={kappa_t}, "
                  f"error={best['kappa_error_pct']}%)")
        else:
            print(f"    -> No feasible design found within budget.")
            best = {"error": "no feasible design"}

        results[name] = best

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(records: List[Dict], records_p6: List[Dict]) -> Dict:
    """Run all P8 hardware design analyses."""
    c_arch_results = analyze_c_arch(records)
    return {
        "c_arch":          c_arch_results,
        "memory_design":   analyze_memory_design(c_arch_results),
        "hw_efficiency":   analyze_hw_efficiency(records),
        "pruning":         analyze_pruning(records_p6),
        "inverse_design":  analyze_inverse_design(c_arch_results),
    }


def main() -> None:
    """Load full.jsonl + results_p6/full.jsonl, run analyze(), save analysis_p8.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="results/full.jsonl")
    p.add_argument("--data_p6", default="results_p6/full.jsonl")
    p.add_argument("--out",     default="results/analysis_p8.json")
    args = p.parse_args()

    records: List[Dict] = []
    with open(args.data) as f:
        for line in f:
            try:    records.append(json.loads(line))
            except: pass
    print(f"Loaded {len(records):,} records (P1-P5)")

    records_p6: List[Dict] = []
    if os.path.exists(args.data_p6):
        with open(args.data_p6) as f:
            for line in f:
                try:    records_p6.append(json.loads(line))
                except: pass
        print(f"Loaded {len(records_p6):,} records (P6)")
    else:
        print(f"P6 data not found at {args.data_p6}, skipping pruning analysis.")

    result = analyze(records, records_p6)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()