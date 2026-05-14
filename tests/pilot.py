# tests/pilot.py
#
# End-to-end smoke test of the full pipeline before a production run.
#
# What it checks
# --------------
# 1. Dataset loading works (CIFAR-10 only, for speed)
# 2. Every pilot arch trains without error
# 3. kappa_input is in a reasonable range [0.001, 1.0]
# 4. kappa_task > 0  (task information is being measured)
# 5. sanity_passed == True  (permuted kappa is below threshold)
# 6. Width Law: kappa * width has CV < 0.3 across two widths
#    (looser threshold than the paper — pilot uses 200 estimator steps)
# 7. Rough time estimate for the full run
#
# Success criteria
# ----------------
# All of the above pass.  Any failure exits with code 1 so CI/CD or a
# calling script can abort before wasting GPU hours.
#
# Usage
#   python tests/pilot.py [--gpu 0] [--data_root ./data]

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from core.config import (
    KAPPA_CFG_PILOT, set_global_seed,
    PILOT_ARCHS, PILOT_WIDTHS, PILOT_DEPTHS,
    PILOT_SEEDS, PILOT_DATASET, PILOT_EPOCHS,
    BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP,
    DATASETS, ARCHS, ACTIVATIONS, WIDTHS, DEPTHS, SEEDS,
    SIGMAS_SWEEP, TEMPS_SWEEP, EPOCH_ARCHS, EPOCH_WIDTHS,
)
from core.dataset import get_loaders, collect_eval, get_n_classes
from core.kappa   import measure_kappa
from models       import build_model

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
log: list = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f"\n       {detail}" if detail else ""))
    log.append((name, cond))
    return cond


def run_pilot(device: torch.device, data_root: str) -> bool:
    """Run the full smoke-test pipeline on a small subset.

    Returns True if all checks pass, False otherwise.
    Exits with code 1 on failure so CI can abort before wasting GPU hours.
    """
    print("=" * 60)
    print(f"Icon_Empirical Pilot  device={device}  dataset={PILOT_DATASET}")
    print("=" * 60)

    # -- data --
    print("\n[1/4] Dataset loading")
    try:
        train_loader, eval_loader = get_loaders(
            PILOT_DATASET, data_root,
            batch_size=BATCH_SIZE, n_eval=KAPPA_CFG_PILOT.n_eval,
        )
        X_raw, X_flat, Y = collect_eval(eval_loader, n=KAPPA_CFG_PILOT.n_eval)
        n_cls = get_n_classes(PILOT_DATASET)
        check("dataset load",   True, f"train={len(train_loader.dataset)}")
        check("eval shape",     X_raw.shape[1:] == (3, 32, 32),
              f"X_raw={tuple(X_raw.shape)}  Y={tuple(Y.shape)}")
    except Exception as e:
        check("dataset load", False, str(e))
        return False

    # -- model training + kappa --
    print("\n[2/4] Model training + kappa measurement")
    pilot_results = []
    total = len(PILOT_ARCHS) * len(PILOT_WIDTHS) * len(PILOT_DEPTHS) * len(PILOT_SEEDS)
    done  = 0

    for arch in PILOT_ARCHS:
        for width in PILOT_WIDTHS:
            for depth in PILOT_DEPTHS:
                for seed in PILOT_SEEDS:
                    done += 1
                    tag  = f"{arch} w={width} d={depth} s={seed}"
                    t0   = time.time()
                    set_global_seed(seed)

                    try:
                        model = build_model(arch, width, depth, "relu",
                                            n_cls, seed=seed).to(device)
                        opt   = torch.optim.AdamW(model.parameters(),
                                                   lr=LR, weight_decay=WEIGHT_DECAY)
                        crit  = nn.CrossEntropyLoss()

                        # short training
                        model.train()
                        for _ in range(PILOT_EPOCHS):
                            for xb, yb in train_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                loss   = crit(model(xb), yb)
                                opt.zero_grad()
                                loss.backward()
                                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                                opt.step()

                        # kappa measurement
                        model.eval()
                        with torch.no_grad():
                            taps: dict = {}
                            model(X_raw.to(device), taps=taps)

                        Z = taps.get("ffn_out")
                        if Z is None:
                            raise ValueError("ffn_out tap missing")

                        res = measure_kappa(
                            X_flat, Z, Y, KAPPA_CFG_PILOT, device,
                            n_classes=n_cls,
                        )
                        elapsed = time.time() - t0
                        pilot_results.append({
                            "arch":         arch,
                            "width":        width,
                            "depth":        depth,
                            "seed":         seed,
                            "kappa_input":  res.kappa_input,
                            "kappa_task":   res.kappa_task,
                            "sanity_passed":res.sanity_passed,
                            "saturated":    res.saturated,
                            "elapsed":      elapsed,
                        })
                        sn = "OK" if res.sanity_passed else "SANITY_FAIL"
                        print(f"  [{done}/{total}] {tag}: "
                              f"ki={res.kappa_input:.4f}  kt={res.kappa_task:.4f}  "
                              f"sat={res.saturated}  {sn}  {elapsed:.0f}s")

                    except Exception as e:
                        print(f"  [{done}/{total}] FAIL {tag}: {e}")
                        log.append((tag, False))

    # -- validation checks --
    print("\n[3/4] Validation")

    ki_vals = [r["kappa_input"] for r in pilot_results]
    if ki_vals:
        in_range = all(0.001 <= k <= 1.0 for k in ki_vals)
        check("kappa_input in [0.001, 1.0]",
              in_range,
              f"min={min(ki_vals):.4f}  max={max(ki_vals):.4f}")

    kt_vals = [r["kappa_task"] for r in pilot_results]
    if kt_vals:
        check("kappa_task > 0",
              all(k > 0 for k in kt_vals),
              f"min={min(kt_vals):.5f}  max={max(kt_vals):.5f}")

    if pilot_results:
        check("sanity_passed all True",
              all(r["sanity_passed"] for r in pilot_results))

    # Width Law with loose CV threshold (pilot has fewer estimator steps)
    print("\n  Width Law (CV < 0.3 with reduced estimator steps):")
    for arch in PILOT_ARCHS:
        for depth in PILOT_DEPTHS:
            kw = {r["width"]: r["kappa_input"]
                  for r in pilot_results
                  if r["arch"] == arch and r["depth"] == depth and r["seed"] == 0}
            widths = sorted(kw.keys())
            if len(widths) >= 2:
                prods  = [w * kw[w] for w in widths]
                mean_c = sum(prods) / len(prods)
                cv     = (max(prods) - min(prods)) / mean_c if mean_c > 0 else 999
                ok     = cv < 0.3
                print(f"  [{PASS if ok else WARN}] {arch} d={depth}: "
                      f"C~={mean_c:.3f}  CV={cv:.3f}")
                log.append((f"width_law {arch} d={depth}", ok))


    # -- time estimate --
    print("\n[4/4] Time estimate")
    if pilot_results:
        from itertools import product as iproduct
        avg_s = sum(r["elapsed"] for r in pilot_results) / len(pilot_results)

        # Replicate the exact condition logic from experiments/runner.py for an accurate count.
        seen = set(); main_conds = []
        for ds, arch, act, w, d, s in iproduct(
                DATASETS, ARCHS, ACTIVATIONS, WIDTHS, DEPTHS, SEEDS):
            if d == 4 or act == "relu":
                key = (ds, arch, act, w, d, s)
                if key not in seen:
                    seen.add(key); main_conds.append(key)

        from core.config import TEMP_SWEEP_WIDTHS, TEMP_SWEEP_DEPTHS
        n_main  = len(main_conds)
        n_sig   = len(ARCHS) * len(SIGMAS_SWEEP) * len(SEEDS) * len(DATASETS)
        # Boltzmann: full grid; Transformer: w=[64,256], d=4 only
        n_temp  = (len(TEMP_SWEEP_WIDTHS) * len(TEMP_SWEEP_DEPTHS)
                   * len(TEMPS_SWEEP) * len(SEEDS) * len(DATASETS)   # boltzmann
                   + 2 * 1 * len(TEMPS_SWEEP) * len(SEEDS) * len(DATASETS))  # transformer w=[64,256]
        n_ep    = (len(EPOCH_ARCHS) * len(ACTIVATIONS) * len(EPOCH_WIDTHS)
                   * len(SEEDS) * len(DATASETS))
        n_total = n_main + n_sig + n_temp + n_ep

        est_h = n_total * avg_s / 8 / 3600

        print(f"  pilot avg:    {avg_s:.1f}s  (est_steps={KAPPA_CFG_PILOT.est_steps})")
        print(f"  main={n_main:,}  sigma={n_sig:,}  temp={n_temp:,}  epoch={n_ep:,}")
        print(f"  total conds:  {n_total:,}")
        print(f"  8-GPU est:    {est_h:.1f} hours")

    # -- final verdict --
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in log if ok)
    total_ = len(log)
    if passed < total_:
        print(f"Pilot FAILED: {passed}/{total_} checks passed")
        for name, ok in log:
            if not ok:
                print(f"  FAIL  {name}")
        return False
    else:
        print(f"Pilot PASSED: {passed}/{total_} checks passed")
        print("Ready to run: bash scripts/run.sh all")
        return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",       type=int, default=0)
    p.add_argument("--data_root", type=str, default="./data")
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    ok = run_pilot(device, args.data_root)
    sys.exit(0 if ok else 1)
