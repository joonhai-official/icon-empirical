# papers/p6_thermo.py
#
# Analysis script for paper sections §7 (Phase Transitions) and §9 (IB Ratio).
#
# Paper mapping
# -------------
#   §7 Phase Transitions          — (w, d) phase diagram, sharp collapse boundaries
#   §9 Information Bottleneck Ratio eta_t = kappa_task / kappa_input
#
#
# Three phenomena investigated:
#
# 1. Fine-grained sigma sweep (sigma_fine)
#    12 points from 0.001 to 10.0.  Resolves the sigma* bell-curve shape
#    with 4x more resolution than P3.  Tests whether sigma* is architecture-
#    specific and whether a universal optimal noise scale exists.
#
# 2. Temperature sweep — extended range (temp_fine)
#    7 points from 0.001 to 1000.0, Boltzmann only.
#    P3 found T-invariant at T=0.01..100; extending to T=1000 tests whether
#    T-sensitivity emerges at extreme temperatures (thermodynamic limit).
#    Expected: kappa drops at T >> 1 as thermal noise overwhelms structure.
#    Connection to Helmholtz free energy F = E - TS: high T drives TS >> E,
#    increasing disorder and reducing information capacity.
#
# 3. Phase transition search (phase)
#    Fine-grained depth sweep d=1..16 for serial and transformer_postln.
#    Both are depth-sensitive; scanning at unit depth steps locates the
#    critical depth d_c where kappa collapses.
#    serial: residual-free chain — expected hard collapse (1st-order-like)
#    transformer_postln: Post-LN — expected soft collapse (continuous)
#
# Usage
#   python papers/p6_thermo.py --data results_p6/full.jsonl \
#                              [--out results_p6/analysis_p6.json]

import argparse
import os
import sys
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Fine-grained sigma sweep
# ---------------------------------------------------------------------------

def analyze_sigma_fine(records: List[Dict]) -> Dict:
    """Find sigma* with 12-point resolution for all architectures.

    Uses exp_type==sigma_fine records.  Seeds averaged per (arch, sigma).
    Reports peak location, curve shape, and arch-to-arch sigma* variation.
    """
    recs = [r for r in records
            if r.get("exp_type") == "sigma_fine"
            and r.get("kappa_input") is not None
            and r.get("sanity_passed", True)]

    print(f"\n=== Fine sigma sweep ({len(recs)} records) ===")
    results: Dict[str, Dict] = {}

    by_key: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = f"{r['arch']}|{r['dataset']}"
        by_key[key][r["sigma"]].append(r["kappa_input"])

    for key in sorted(by_key):
        arch, ds = key.split("|")
        by_sigma  = by_key[key]
        sigmas    = sorted(by_sigma)
        kappas    = [sum(by_sigma[s]) / len(by_sigma[s]) for s in sigmas]

        sigma_star_idx = kappas.index(max(kappas))
        sigma_star     = sigmas[sigma_star_idx]

        # CV across sigma range (measures sensitivity)
        mean_k = sum(kappas) / len(kappas)
        cv     = (max(kappas) - min(kappas)) / mean_k if mean_k > 0 else 0.0

        # Bell-curve test: peak not at boundary
        is_bell = 0 < sigma_star_idx < len(sigmas) - 1

        print(f"\n  {arch} ({ds})  sigma*={sigma_star}  CV={cv:.4f}  bell={is_bell}")
        for s, k in zip(sigmas, kappas):
            marker = "  <- sigma*" if s == sigma_star else ""
            print(f"    sigma={s:<6}  kappa={k:.5f}{marker}")

        results[key] = {
            "sigma_star":          sigma_star,
            "kappa_at_sigma_star": round(max(kappas), 6),
            "cv":                  round(cv, 6),
            "is_bell_curve":       is_bell,
            "curve":               {str(s): round(k, 6) for s, k in zip(sigmas, kappas)},
        }

    return results


# ---------------------------------------------------------------------------
# 2. Extended temperature sweep
# ---------------------------------------------------------------------------

def analyze_temp_fine(records: List[Dict]) -> Dict:
    """Extended T sweep: 0.001..1000, Boltzmann only.

    kappa_input = I(X; Z_tilde(T)) / d_z
    T* = argmax_T kappa(T)
    CV = (max-min)/mean across T values

    Connection to thermodynamics:
      High T -> large TS term in F=E-TS -> disorder dominates -> kappa drops
      T* is the optimal temperature balancing structure and thermal noise.
    """
    recs = [r for r in records
            if r.get("exp_type") == "temp_fine"
            and r.get("kappa_input") is not None
            and r.get("sanity_passed")]

    print(f"\n=== Extended T sweep ({len(recs)} records) ===")
    results: Dict[str, Dict] = {}

    # Group by (arch, width, depth, dataset)
    by_cfg: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = f"{r['arch']}|w{r['width']}|d{r['depth']}|{r['dataset']}"
        by_cfg[key][r["temperature"]].append(r["kappa_input"])

    for key in sorted(by_cfg):
        by_T   = by_cfg[key]
        temps  = sorted(by_T)
        kappas = [sum(by_T[T]) / len(by_T[T]) for T in temps]

        mean_k = sum(kappas) / len(kappas)
        cv     = (max(kappas) - min(kappas)) / mean_k if mean_k > 0 else 0.0
        t_inv  = cv < 0.05

        # Find T* and T_collapse (where kappa drops >10% from max)
        t_star_idx = kappas.index(max(kappas))
        t_star     = temps[t_star_idx]
        k_max      = kappas[t_star_idx]
        t_collapse: Optional[float] = None
        for i, (T, k) in enumerate(zip(temps, kappas)):
            if T > t_star and k < 0.9 * k_max:
                t_collapse = T
                break

        label = "T-invariant" if t_inv else "T-sensitive"
        print(f"\n  {key}  CV={cv:.4f}  T*={t_star}  T_collapse={t_collapse}  {label}")
        for T, k in zip(temps, kappas):
            marker = "  <- T*" if T == t_star else ""
            print(f"    T={T:<8}  kappa={k:.5f}{marker}")

        results[key] = {
            "cv":          round(cv, 6),
            "t_invariant": t_inv,
            "t_star":      t_star,
            "t_collapse":  t_collapse,
            "curve":       {str(T): round(k, 6) for T, k in zip(temps, kappas)},
        }

    return results


# ---------------------------------------------------------------------------
# 3. Phase transition
# ---------------------------------------------------------------------------

def _find_critical_depth(depths: List[int],
                         kappas: List[float],
                         threshold: float = 0.1) -> Optional[int]:
    """Return the first depth where kappa drops below threshold * kappa_d1.

    kappa_d1 is the value at depth=1 (reference).
    Returns None if no collapse is found.
    """
    if not kappas or kappas[0] == 0:
        return None
    ref = kappas[0]
    for d, k in zip(depths, kappas):
        if k < threshold * ref:
            return d
    return None


def analyze_phase(records: List[Dict]) -> Dict:
    """Locate critical depth d_c where kappa collapses (phase transition).

    kappa_input = I(X; Z_tilde) / d_z measured at d=1..16 (unit steps).
    d_c = first depth where kappa_input < 0.1 * kappa_input(d=1).

    serial:              hard collapse (residual-free) — 1st-order-like
    transformer_postln:  soft collapse (Post-LN instability) — continuous

    Also measures d_c for kappa_task separately, which may differ.
    """
    recs = [r for r in records
            if r.get("exp_type") == "phase"
            and r.get("kappa_input") is not None
            and r.get("sanity_passed")]

    print(f"\n=== Phase transition search ({len(recs)} records) ===")
    results: Dict[str, Dict] = {}

    # Group by (arch, width, dataset), average over seeds
    by_key: Dict[str, Dict[int, Tuple[List, List]]] = defaultdict(
        lambda: defaultdict(lambda: ([], [])))
    for r in recs:
        key = f"{r['arch']}|w{r['width']}|{r['dataset']}"
        by_key[key][r["depth"]][0].append(r["kappa_input"])
        by_key[key][r["depth"]][1].append(r.get("kappa_task") or 0.0)

    for key in sorted(by_key):
        arch = key.split("|")[0]
        by_d = by_key[key]
        depths  = sorted(by_d)
        ki_vals = [sum(by_d[d][0]) / len(by_d[d][0]) for d in depths]
        kt_vals = [sum(by_d[d][1]) / len(by_d[d][1]) for d in depths]

        d_c_ki = _find_critical_depth(depths, ki_vals, threshold=0.1)
        d_c_kt = _find_critical_depth(depths, kt_vals, threshold=0.1)

        # Collapse type: hard (kappa reaches exactly 0) vs soft
        min_ki     = min(ki_vals)
        is_hard    = min_ki < 1e-5
        collapse   = "hard" if is_hard else "soft"

        print(f"\n  {key}")
        print(f"    d_c(kappa_input)={d_c_ki}  d_c(kappa_task)={d_c_kt}  collapse={collapse}")
        print(f"    {'depth':>6}  {'ki':>10}  {'kt':>10}")
        for d, ki, kt in zip(depths, ki_vals, kt_vals):
            marker = " <- d_c" if d == d_c_ki else ""
            print(f"    {d:>6}  {ki:>10.5f}  {kt:>10.5f}{marker}")

        results[key] = {
            "d_c_kappa_input":  d_c_ki,
            "d_c_kappa_task":   d_c_kt,
            "collapse_type":    collapse,
            "ki_at_d1":         round(ki_vals[0], 6) if ki_vals else None,
            "ki_min":           round(min_ki, 6),
            "profile_ki":       {str(d): round(k, 6) for d, k in zip(depths, ki_vals)},
            "profile_kt":       {str(d): round(k, 6) for d, k in zip(depths, kt_vals)},
        }

    # Summary: d_c by arch
    print("\n--- Phase transition summary ---")
    dc_by_arch: Dict[str, List] = defaultdict(list)
    for key, v in results.items():
        arch = key.split("|")[0]
        if v["d_c_kappa_input"] is not None:
            dc_by_arch[arch].append(v["d_c_kappa_input"])
    for arch, dcs in sorted(dc_by_arch.items()):
        mean_dc = sum(dcs) / len(dcs)
        print(f"  {arch:25s}  d_c(mean)={mean_dc:.1f}  values={dcs}")

    results["_summary"] = {
        arch: {"mean_d_c": round(sum(dcs)/len(dcs), 2), "all_d_c": dcs}
        for arch, dcs in dc_by_arch.items()
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Run all P6 analyses: sigma_fine, temp_fine, phase transition."""
    return {
        "sigma_fine": analyze_sigma_fine(records),
        "temp_fine":  analyze_temp_fine(records),
        "phase":      analyze_phase(records),
    }


def main() -> None:
    """Load full.jsonl (P6 experiments), run analyze(), save analysis_p6.json.

    --data     : path to P6 full.jsonl (default: results_p6/full.jsonl)
    --data_p6  : alias for --data, accepted for interface consistency with p8_hw.py
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="results_p6/full.jsonl",
                   help="P6 experiment records (results_p6/full.jsonl)")
    p.add_argument("--data_p6", default=None,
                   help="Alias for --data; overrides --data if provided")
    p.add_argument("--out",     default="results_p6/analysis_p6.json")
    args = p.parse_args()

    data_path = args.data_p6 if args.data_p6 is not None else args.data

    records: List[Dict] = []
    with open(data_path) as f:
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
