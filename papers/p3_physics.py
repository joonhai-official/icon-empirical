# papers/p3_physics.py
#
# Analysis script for paper appendices B (sigma sweep) and C (temperature sweep).
#
# Paper mapping
# -------------
#   App. B sigma-Sweep (Robustness to Measurement Perturbation)
#   App. C Temperature Sweep (Boltzmann and Pre-LN)
#
#
# sigma sweep
# -----------
# We measure kappa at sigma = 0.01 .. 1.0 with all other variables fixed
# (width=256, depth=4).  The goal is to find sigma* = argmax_sigma kappa(sigma).
#
# Theory: sigma acts as a regulariser in the InfoNCE estimator.  Too small
# -> critic overfits, MI overestimated (high variance).  Too large -> noise
# drowns the signal, MI underestimated.  sigma* is the sweet spot.
# If sigma* is consistent across architectures it suggests an intrinsic noise
# scale for neural representations.
#
# Temperature sweep (Boltzmann + Transformer — P3 core comparison)
# -----------------------------------------------------------------
# Only architectures with explicit softmax-like gating carry a natural
# temperature interpretation:
#   Boltzmann    : h = sigmoid(v @ W / T)   — mean-field RBM
#   Transformer  : a = softmax(q k^T / (sqrt(d) * T)) — scaled dot-product
#
# Each T value is an independent run: model trained and evaluated at the same T.
# This tests whether the learned representation quality varies with T,
# i.e. whether T affects what the model can represent (not just at eval).
#
# Expected outcome:
#   Boltzmann T-sensitive  (CV >= 0.05): sigmoid entropy directly controlled by T
#   Transformer T-invariant (CV < 0.05): learned W_q, W_k compensate for T
#
# This contrast is the P3 physics claim: attention representations are
# thermodynamically robust; energy-based representations are not.
#
# MaxEnt efficiency
# -----------------
# The InfoNCE upper bound gives kappa_max = log(batch_size) / d_z.
# Efficiency eta = kappa / kappa_max in [0, 1] measures how fully the
# representation exploits its available capacity.
# eta stable across widths -> C_arch captures the architecture effect cleanly.
#
# Usage
#   python papers/p3_physics.py --data results/full.jsonl [--out results/analysis_p3.json]

import argparse
import os
import sys
import json
import math
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.config import TEMP_SWEEP_ARCHS as _EXPECTED_T_ARCHS
except ImportError:
    _EXPECTED_T_ARCHS = ["boltzmann", "transformer_preln"]


# ---------------------------------------------------------------------------
# sigma sweep
# ---------------------------------------------------------------------------

def analyze_sigma(records: List[Dict]) -> Dict:
    """Find σ* = argmax_σ κ(σ) for each architecture.

    Records with exp_type==sigma and not saturated only.
    """
    sigma_recs = [r for r in records
                  if r.get("exp_type") == "sigma"
                  and r.get("kappa_input") is not None
                  and r.get("sanity_passed")]

    print(f"\n=== sigma sweep ({len(sigma_recs)} records) ===")
    results: Dict[str, Dict] = {}

    # group by (arch, dataset), average over seeds and activations
    by_key: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in sigma_recs:
        key = f"{r['arch']}|{r['dataset']}"
        by_key[key][r["sigma"]].append(r["kappa_input"])

    for key in sorted(by_key):
        arch, ds = key.split("|")
        by_sigma = by_key[key]
        sigmas   = sorted(by_sigma)
        kappas   = [sum(by_sigma[s]) / len(by_sigma[s]) for s in sigmas]

        # find sigma*
        sigma_star_idx = kappas.index(max(kappas))
        sigma_star     = sigmas[sigma_star_idx]

        # monotone decreasing test
        is_mono_dec = all(kappas[i] >= kappas[i + 1] for i in range(len(kappas) - 1))

        print(f"\n  {arch} ({ds})  sigma*={sigma_star}  mono_dec={is_mono_dec}")
        for s, k in zip(sigmas, kappas):
            marker = "  <- sigma*" if s == sigma_star else ""
            print(f"    sigma={s:.3f}  kappa={k:.5f}{marker}")

        results[key] = {
            "sigma_star":           sigma_star,
            "kappa_at_sigma_star":  round(max(kappas), 6),
            "is_monotone_dec":      is_mono_dec,
            "curve":                {str(s): round(k, 6) for s, k in zip(sigmas, kappas)},
        }

    return results


# ---------------------------------------------------------------------------
# Temperature sweep (Boltzmann)
# ---------------------------------------------------------------------------

def analyze_temperature(records: List[Dict]) -> Dict:
    """T-invariance test: Boltzmann (expected sensitive) vs Transformer (expected invariant).

    CV = (max-min)/mean of kappa across T values.
    CV >= 0.05 → T-sensitive; CV < 0.05 → T-invariant.
    """
    temp_recs = [r for r in records
                 if r.get("exp_type") == "temp"
                 and r.get("kappa_input") is not None
                 and r.get("sanity_passed")]

    print(f"\n=== T sweep ({len(temp_recs)} records) ===")
    results: Dict[str, Dict] = {}

    # group by (arch, activation, width, depth, dataset)
    by_cfg: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in temp_recs:
        key = f"{r['arch']}|{r['activation']}|w{r['width']}|d{r['depth']}|{r['dataset']}"
        by_cfg[key][r["temperature"]].append(r["kappa_input"])

    arch_verdicts: Dict[str, List[bool]] = defaultdict(list)

    for key in sorted(by_cfg):
        by_T   = by_cfg[key]
        temps  = sorted(by_T)
        kappas = [sum(by_T[T]) / len(by_T[T]) for T in temps]

        mean_k = sum(kappas) / len(kappas)
        std_k  = math.sqrt(sum((k - mean_k) ** 2 for k in kappas) / len(kappas))
        cv     = std_k / mean_k if mean_k > 0 else 0.0

        t_inv = cv < 0.05
        label = "T-invariant" if t_inv else "T-sensitive"
        arch  = key.split("|")[0]
        arch_verdicts[arch].append(t_inv)

        print(f"\n  {key}  CV={cv:.4f}  {label}")
        for T, k in zip(temps, kappas):
            print(f"    T={T:7.2f}  kappa={k:.5f}")

        results[key] = {
            "arch":        arch,
            "cv":          round(cv, 6),
            "t_invariant": t_inv,
            "curve":       {str(T): round(k, 6) for T, k in zip(temps, kappas)},
        }

    # per-arch summary: fraction of T-invariant configurations
    # Cross-check that the expected architectures actually appear in the data
    print("\n--- T sweep summary by architecture ---")
    for expected in _EXPECTED_T_ARCHS:
        if expected not in arch_verdicts:
            print(f"  WARNING: expected arch '{expected}' not found in temp records")
    arch_summary: Dict[str, Dict] = {}
    for arch, verdicts in sorted(arch_verdicts.items()):
        inv_frac = sum(verdicts) / len(verdicts) if verdicts else 0.0
        overall  = "T-invariant" if inv_frac >= 0.8 else "T-sensitive"
        print(f"  {arch:25s}  T-invariant={sum(verdicts)}/{len(verdicts)}"
              f"  → {overall}")
        arch_summary[arch] = {
            "t_invariant_fraction": round(inv_frac, 4),
            "overall_verdict":      overall,
        }
    results["_arch_summary"] = arch_summary

    return results


# ---------------------------------------------------------------------------
# MaxEnt efficiency
# ---------------------------------------------------------------------------

def analyze_maxent(records: List[Dict]) -> Dict:
    """
    kappa_max = log(512) / d_z  (InfoNCE upper bound divided by dimension)
    eta       = kappa_input / kappa_max

    If eta is stable across widths within an arch, C_arch is a clean
    architecture constant independent of any capacity artefact.
    """
    print("\n=== MaxEnt efficiency: eta = kappa / kappa_max ===")
    log_batch = math.log(512)

    # Exclude saturated records: their kappa ≈ kappa_max by definition,
    # making eta ≈ 1 regardless of the true representation efficiency.
    main = [r for r in records
            if r.get("exp_type") == "main"
            and r.get("depth") == 4
            and r.get("kappa_input") is not None
            and r.get("d_z") is not None
            and r.get("sanity_passed")]

    results: Dict[str, Dict] = {}
    by_arch_w: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
    for r in main:
        by_arch_w[r["arch"]][r["width"]].append(r)

    for arch in sorted(by_arch_w):
        print(f"\n  {arch}")
        etas = []
        for w in sorted(by_arch_w[arch]):
            rs    = by_arch_w[arch][w]
            ki    = sum(r["kappa_input"] for r in rs) / len(rs)
            d_z   = rs[0]["d_z"]
            k_max = log_batch / d_z
            eta   = ki / k_max if k_max > 0 else 0.0
            etas.append(eta)
            print(f"    w={w:4d}  kappa={ki:.5f}  kappa_max={k_max:.5f}  eta={eta:.4f}")
            results[f"{arch}|w{w}"] = {
                "kappa_input": round(ki, 6),
                "kappa_max":   round(k_max, 6),
                "eta":         round(eta, 6),
                "d_z":         d_z,
            }

        if len(etas) >= 2:
            mean_eta = sum(etas) / len(etas)
            cv_eta   = (max(etas) - min(etas)) / mean_eta if mean_eta > 0 else 0.0
            stable   = cv_eta < 0.1
            print(f"    eta CV={cv_eta:.4f}  {'stable' if stable else 'varies'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Run sigma sweep, T sweep, and MaxEnt efficiency analysis.

    Returns combined dict with sigma, temperature, and maxent keys.
    """
    return {
        "sigma":       analyze_sigma(records),
        "temperature": analyze_temperature(records),
        "maxent":      analyze_maxent(records),
    }


def main() -> None:
    """Load full.jsonl, run analyze(), save results/analysis_p3.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/full.jsonl")
    p.add_argument("--out",  default="results/analysis_p3.json")
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
