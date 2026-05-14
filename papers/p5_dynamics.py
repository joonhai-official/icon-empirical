# papers/p5_dynamics.py
#
# Analysis script for paper section §8 (Training Dynamics).
#
# Paper mapping
# -------------
#   §8 Training Dynamics — kappa_input and kappa_task trajectories across epochs
#
#
# Definitions
# -----------
#   kappa_input(epoch) = I(X; Z_d(epoch)) / d_z
#   kappa_task(epoch)  = I(Y; Z_d(epoch)) / d_z
# Both measured at the ffn_out tap (canonical d_z = width).
#
# The Information Bottleneck hypothesis (Shwartz-Ziv & Tishby 2017) predicts
# two phases during SGD:
#   fitting phase     : kappa_input rises as the network learns input structure
#   compression phase : kappa_input drops as irrelevant structure is discarded
#
# Saxe et al. (2018) showed this depends on the activation function — relu
# networks often skip compression entirely.  We test this with three
# activations and measure both kappa_input and kappa_task trajectories.
#
# Pattern taxonomy
# ----------------
#   u_shape       : initial drop (>=0.5%) then recovery (>=0.5%) — IB two-phase
#   monotone_inc  : kappa rises throughout — fitting only, no compression
#   monotone_dec  : kappa falls throughout — compression only
#   fluctuating   : neither monotone nor U-shaped
#
# Threshold is 0.5% rather than the conventional 2% because kappa_epoch
# values cluster tightly (~2% total range); a 2% guard misses genuine
# U-shapes in real measurements.
#
# kappa vs accuracy correlation
# ------------------------------
# Having test_acc at every checkpoint (epoch_kappas["test_acc"]) lets us
# compute Pearson r between kappa_task(epoch) and test_acc(epoch).
# r > 0.8 would support using kappa_task as a training-time proxy for
# final model quality — the key claim connecting Icon_Empirical to practical design.
#
# Usage
#   python papers/p5_dynamics.py --data results/full.jsonl [--out results/analysis_p5.json]

import argparse
import os
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def detect_pattern(epochs: List[int], kappas: List[float]) -> Dict:
    """
    Classify a kappa(epoch) trajectory into one of four patterns.

    U-shape condition: the minimum is strictly interior (not at endpoint),
    the drop from start to minimum is >= 0.5%, and the recovery from minimum
    to end is >= 0.5%.
    """
    if len(epochs) < 3:
        return {"pattern": "insufficient_data", "phase_change_epoch": None}

    n       = len(kappas)
    min_idx = kappas.index(min(kappas))

    drop     = (kappas[0] - kappas[min_idx]) / kappas[0] if kappas[0] > 0 else 0
    recovery = (kappas[-1] - kappas[min_idx]) / kappas[-1] if kappas[-1] > 0 else 0
    u_shape  = (0 < min_idx < n - 1 and drop > 0.005 and recovery > 0.005)

    deltas   = [kappas[i + 1] - kappas[i] for i in range(n - 1)]
    mono_inc = all(d >= -1e-4 for d in deltas)
    mono_dec = all(d <=  1e-4 for d in deltas)

    if u_shape:    pattern = "u_shape"
    elif mono_inc: pattern = "monotone_inc"
    elif mono_dec: pattern = "monotone_dec"
    else:          pattern = "fluctuating"

    return {
        "pattern":           pattern,
        "phase_change_epoch": epochs[min_idx] if u_shape else None,
        "initial_kappa":     round(kappas[0],  6),
        "min_kappa":         round(min(kappas), 6),
        "final_kappa":       round(kappas[-1], 6),
        "total_change_pct":  round((kappas[-1] - kappas[0]) / kappas[0] * 100, 2)
                             if kappas[0] > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def pearson_r(xs: List[float], ys: List[float]) -> float:
    """
    Pearson correlation coefficient between two equal-length sequences.
    Returns 0.0 if either sequence has zero variance.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return round(num / (sx * sy), 6)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(records: List[Dict]) -> Dict:
    """Learning dynamics analysis: IB two-phase patterns + κ_task vs accuracy.

    Epoch records are NOT saturated-filtered — the saturation phase is
    itself an object of study in the trajectory analysis.
    """
    ep_recs = [r for r in records
               if r.get("exp_type") == "epoch"
               and r.get("epoch_kappas")
               and len(r["epoch_kappas"]) > 0]
    # Note: epoch records are NOT filtered by saturated flag.
    # The full trajectory (including saturation phase) is the
    # object of study.  Individual saturated epoch_kappas entries
    # are identified by their kappa values approaching kappa_max.

    print(f"\n=== kappa_epoch analysis ({len(ep_recs)} conditions) ===")
    out: Dict = {}
    all_analyses: Dict[str, Dict] = {}
    pattern_counter: Dict[str, int] = defaultdict(int)

    by_cfg: Dict[str, List] = defaultdict(list)
    for r in ep_recs:
        key = f"{r['arch']}|w{r['width']}|{r['activation']}|{r['dataset']}"
        by_cfg[key].append(r)

    for key in sorted(by_cfg):
        rs = by_cfg[key]

        ki_ep:   Dict[int, List[float]] = defaultdict(list)
        kt_ep:   Dict[int, List[float]] = defaultdict(list)
        acc_ep:  Dict[int, List[float]] = defaultdict(list)

        for r in rs:
            for entry in r["epoch_kappas"]:
                e = entry["epoch"]
                if entry.get("kappa_input") is not None:
                    ki_ep[e].append(entry["kappa_input"])
                if entry.get("kappa_task") is not None:
                    kt_ep[e].append(entry["kappa_task"])
                if entry.get("test_acc") is not None:
                    acc_ep[e].append(entry["test_acc"])

        epochs   = sorted(ki_ep)
        ki_means = [sum(ki_ep[e]) / len(ki_ep[e]) for e in epochs]
        kt_means = [sum(kt_ep[e]) / len(kt_ep[e]) for e in epochs if e in kt_ep]
        ac_means = [sum(acc_ep[e]) / len(acc_ep[e]) for e in epochs if e in acc_ep]

        ki_info = detect_pattern(epochs, ki_means)
        pattern_counter[ki_info["pattern"]] += 1

        # kappa_task vs test_acc correlation (§8 core claim).
        # Use only epochs that have both kappa_task AND test_acc to avoid
        # index misalignment when some entries are missing one value.
        r_kt_acc = 0.0
        r_ki_acc = 0.0
        shared_epochs_kt  = sorted(set(kt_ep) & set(acc_ep))
        shared_epochs_ki  = sorted(set(ki_ep) & set(acc_ep))
        if shared_epochs_kt:
            kt_shared  = [sum(kt_ep[e])  / len(kt_ep[e])  for e in shared_epochs_kt]
            acc_shared_kt = [sum(acc_ep[e]) / len(acc_ep[e]) for e in shared_epochs_kt]
            r_kt_acc = pearson_r(kt_shared, acc_shared_kt)
        if shared_epochs_ki:
            ki_shared  = [sum(ki_ep[e])  / len(ki_ep[e])  for e in shared_epochs_ki]
            acc_shared_ki = [sum(acc_ep[e]) / len(acc_ep[e]) for e in shared_epochs_ki]
            r_ki_acc = pearson_r(ki_shared, acc_shared_ki)

        print(f"\n  {key}")
        print(f"  pattern={ki_info['pattern']}  "
              f"phase_change_epoch={ki_info['phase_change_epoch']}")
        print(f"  r(kappa_task, acc)={r_kt_acc:.3f}  "
              f"r(kappa_input, acc)={r_ki_acc:.3f}")
        print(f"  {'ep':>4}  {'ki':>8}  {'kt':>8}  {'acc':>7}")
        for i, ep in enumerate(epochs):
            kt  = kt_means[i] if i < len(kt_means) else 0.0
            acc = ac_means[i] if i < len(ac_means) else 0.0
            print(f"  {ep:>4}  {ki_means[i]:>8.5f}  {kt:>8.5f}  {acc:>7.4f}")

        all_analyses[key] = {
            "ki_info":   ki_info,
            "epochs":    epochs,
            "ki_means":  [round(k, 6) for k in ki_means],
            "kt_means":  [round(k, 6) for k in kt_means],
            "acc_means": [round(a, 6) for a in ac_means],
            "r_kappa_task_vs_acc":  r_kt_acc,
            "r_kappa_input_vs_acc": r_ki_acc,
        }

    out["analyses"] = all_analyses

    # Pattern summary
    print("\n=== Pattern counts ===")
    for pat, cnt in sorted(pattern_counter.items()):
        print(f"  {pat}: {cnt}")
    out["pattern_counts"] = dict(pattern_counter)

    # Activation breakdown
    print("\n=== Activation vs pattern ===")
    act_results: Dict[str, Dict] = {}
    for act in ["relu", "gelu", "tanh"]:
        pats = [v["ki_info"]["pattern"]
                for k, v in all_analyses.items()
                if f"|{act}|" in k]
        u_count = pats.count("u_shape")
        print(f"  {act}  u_shape={u_count}/{len(pats)}")
        act_results[act] = {"u_shape_count": u_count, "total": len(pats)}
    out["by_activation"] = act_results

    # kappa_task vs accuracy — aggregate across all conditions
    print("\n=== kappa_task vs test_acc (§8 core finding) ===")
    all_r_kt = [v["r_kappa_task_vs_acc"]
                for v in all_analyses.values()
                if v["r_kappa_task_vs_acc"] != 0.0]
    all_r_ki = [v["r_kappa_input_vs_acc"]
                for v in all_analyses.values()
                if v["r_kappa_input_vs_acc"] != 0.0]

    if all_r_kt:
        mean_r_kt = sum(all_r_kt) / len(all_r_kt)
        mean_r_ki = sum(all_r_ki) / len(all_r_ki) if all_r_ki else 0.0
        print(f"  mean r(kappa_task, acc)  = {mean_r_kt:.3f}  "
              f"({'supported' if mean_r_kt > 0.8 else 'weak'}  target > 0.8)")
        print(f"  mean r(kappa_input, acc) = {mean_r_ki:.3f}")
        out["kappa_vs_acc"] = {
            "mean_r_kappa_task":  round(mean_r_kt, 4),
            "mean_r_kappa_input": round(mean_r_ki, 4),
            "n_conditions":       len(all_r_kt),
            "claim_supported":    mean_r_kt > 0.8,
        }

    return out


def main() -> None:
    """Load full.jsonl, run analyze(), save results/analysis_p5.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/full.jsonl")
    p.add_argument("--out",  default="results/analysis_p5.json")
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
