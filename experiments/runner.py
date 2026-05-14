# experiments/runner.py
#
# Unified experiment loop covering all four experiment types:
#   main  — full grid (arch x activation x width x depth x seed x dataset)
#   sigma — sigma sweep at fixed width=256, depth=4
#   temp  — temperature sweep (Boltzmann + Transformer T-invariance comparison)
#   epoch — kappa measured at training checkpoints (learning dynamics)
#   all   — all four concatenated
#
# Condition partitioning
# ----------------------
# The full condition list is enumerated up front and split by index modulo
# n_shards.  Each GPU worker (shard) processes its slice independently with
# no inter-process communication.  scripts/run.sh launches one process per GPU.
#
# Main grid design
# ----------------
# Two papers need different sweeps from the same training run:
#   Paper 1 (Width Law):  all activations, depth=4 fixed
#   Paper 2 (Depth Law):  relu only, all depths
# The union covers both without running gelu/tanh at every depth, which
# would nearly triple the condition count for no scientific gain.
#   Rule: include if  depth == 4  OR  activation == "relu"
#
# Resume
# ------
# Completed conditions are identified by a 9-tuple key read from the output
# JSONL.  Restart simply skips any key already present — no re-running.
#
# Error handling
# --------------
# Exceptions in run_one() are written as error records; the loop continues.
# A single OOM at depth=32/width=1024 does not abort the other 5,129 jobs.
#
# Output: JSONL — one JSON object per line, one file per shard.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import time
import traceback
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from core.config import (
    DATASETS, ARCHS, ACTIVATIONS, WIDTHS, DEPTHS, SEEDS,
    SIGMA_CANONICAL, SIGMAS_SWEEP, SIGMA_SWEEP_WIDTH, SIGMA_SWEEP_DEPTH,
    TEMP_CANONICAL, TEMPS_SWEEP, TEMP_SWEEP_WIDTHS, TEMP_SWEEP_DEPTHS,
    EPOCH_ARCHS, EPOCH_WIDTHS, EPOCH_DEPTHS, EPOCH_CHECKPOINTS,
    TRAIN_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP,
    KAPPA_CFG, N_SHARDS,
    KappaCfg,
    set_global_seed,
)
from core.dataset import get_loaders, collect_eval, get_n_classes
from core.kappa   import measure_kappa, measure_layerwise, compute_d_effective
from models       import build_model


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------

def _base(exp_type: str, **kw) -> Dict:
    return {"exp_type": exp_type,
            "sigma":    SIGMA_CANONICAL,
            "temperature": TEMP_CANONICAL,
            **kw}


def main_conditions() -> List[Dict]:
    """Generate the main experiment condition list (4,032 conditions).

    Union rule: depth==4 (all activations, P1) OR activation==relu (all depths, P2).
    Deduplication via a seen-set prevents double-counting the overlap.
    """
    # Paper 1 needs all activations at depth=4.
    # Paper 2 needs all depths at relu only.
    # Union of both rules covers both papers without redundant conditions.
    seen: Set[tuple] = set()
    conds: List[Dict] = []
    for ds, arch, act, w, d, s in product(
            DATASETS, ARCHS, ACTIVATIONS, WIDTHS, DEPTHS, SEEDS):
        if d == 4 or act == "relu":
            key = (ds, arch, act, w, d, s)
            if key not in seen:
                seen.add(key)
                conds.append(_base("main",
                    dataset=ds, arch=arch, activation=act,
                    width=w, depth=d, seed=s))
    return conds


def sigma_conditions() -> List[Dict]:
    """Generate sigma sweep conditions (504 conditions).

    w=256, d=4, relu fixed so only σ varies; finds σ*=argmax_σ κ(σ).
    """
    # Width and depth are fixed so only sigma varies.
    # relu only — sigma* measures the architecture's intrinsic noise scale,
    # which is a property of the topology, not the activation function.
    # 504 conditions vs 1,512 for all activations; the physics claim does not
    # require activation-level sigma breakdown.
    conds = []
    for ds, arch, s, sigma in product(
            DATASETS, ARCHS, SEEDS, SIGMAS_SWEEP):
        conds.append(_base("sigma",
            dataset=ds, arch=arch, activation="relu",
            width=SIGMA_SWEEP_WIDTH, depth=SIGMA_SWEEP_DEPTH,
            seed=s, sigma=sigma))
    return conds


def temp_conditions() -> List[Dict]:
    """Generate T sweep conditions (630 conditions).

    Boltzmann: full width/depth grid (540 conds) — expected T-sensitive.
    Transformer_preln: w=[64,256], d=4 (90 conds) — expected T-invariant.
    Both train and evaluate at the same T (independent runs per T value).
    """
    # T sweep covers two architectures for direct P3 comparison:
    #   boltzmann        : sigmoid T → expected T-sensitive
    #   transformer_preln: attention softmax T → expected T-invariant
    #
    # Boltzmann: full width/depth grid to show T-sensitivity robustly
    # Transformer: representative subset (w=[64,256], d=4) — depth-invariant
    #   so d=4 is sufficient; smaller grid keeps total runtime manageable.
    conds = []
    # Boltzmann: full grid
    for ds, w, d, s, T in product(
            DATASETS, TEMP_SWEEP_WIDTHS, TEMP_SWEEP_DEPTHS, SEEDS, TEMPS_SWEEP):
        conds.append(_base("temp",
            dataset=ds, arch="boltzmann", activation="relu",
            width=w, depth=d, seed=s, temperature=T))
    # Transformer: representative subset
    for ds, w, s, T in product(
            DATASETS, [64, 256], SEEDS, TEMPS_SWEEP):
        conds.append(_base("temp",
            dataset=ds, arch="transformer_preln", activation="relu",
            width=w, depth=4, seed=s, temperature=T))
    return conds


def epoch_conditions() -> List[Dict]:
    """Generate epoch checkpoint conditions (54 conditions).

    transformer_preln only (clearest IB trajectory signal).
    kappa measured at epochs [1,2,5,10,20,50,100,200] per condition.
    """
    # transformer_preln only — shows the clearest learning dynamics signal.
    conds = []
    for ds, arch, act, w, d, s in product(
            DATASETS, EPOCH_ARCHS, ACTIVATIONS,
            EPOCH_WIDTHS, EPOCH_DEPTHS, SEEDS):
        conds.append(_base("epoch",
            dataset=ds, arch=arch, activation=act,
            width=w, depth=d, seed=s))
    return conds



# ---------------------------------------------------------------------------
# P6 Thermo condition generators
# ---------------------------------------------------------------------------

def sigma_fine_conditions() -> List[Dict]:
    """Fine-grained sigma sweep: 12 points, 8 archs (P6).
    Resolves sigma* bell-curve shape with 4x more resolution than P3.
    288 conditions total.
    """
    from core.config import SIGMAS_FINE
    conds = []
    for ds, arch, s, sigma in product(
            DATASETS, ARCHS, SEEDS, SIGMAS_FINE):
        conds.append(_base("sigma_fine",
            dataset=ds, arch=arch, activation="relu",
            width=SIGMA_SWEEP_WIDTH, depth=SIGMA_SWEEP_DEPTH,
            seed=s, sigma=sigma))
    return conds


def temp_fine_conditions() -> List[Dict]:
    """Fine-grained T sweep: 7 points, Boltzmann only (P6).
    100-epoch training tests whether T-invariance persists with full convergence.
    63 conditions total.
    """
    from core.config import TEMPS_FINE, THERMO_TRAIN_EPOCHS
    conds = []
    for ds, w, s, T in product(
            DATASETS, TEMP_SWEEP_WIDTHS, SEEDS, TEMPS_FINE):
        conds.append(_base("temp_fine",
            dataset=ds, arch="boltzmann", activation="relu",
            width=w, depth=4, seed=s, temperature=T))
    return conds


def phase_conditions() -> List[Dict]:
    """Fine-grained depth sweep for phase transition search (P6).
    serial and transformer_postln at d=1..16 unit steps.
    Locates critical depth d_c where kappa collapses.
    468 conditions total.
    """
    from core.config import DEPTHS_FINE, PHASE_ARCHS
    conds = []
    for ds, arch, w, d, s in product(
            DATASETS, PHASE_ARCHS, WIDTHS, DEPTHS_FINE, SEEDS):
        conds.append(_base("phase",
            dataset=ds, arch=arch, activation="relu",
            width=w, depth=d, seed=s))
    return conds


def epoch_fine_conditions() -> List[Dict]:
    """Extended learning dynamics: mlp+cnn+transformer, 100 epochs (P4 booster).
    54 conditions total.
    """
    conds = []
    for ds, arch, act, w, s in product(
            DATASETS, ["mlp", "cnn", "transformer_preln"],
            ACTIVATIONS, [64, 256], SEEDS):
        conds.append(_base("epoch_fine",
            dataset=ds, arch=arch, activation=act,
            width=w, depth=4, seed=s))
    return conds


def get_conditions(exp: str) -> List[Dict]:
    """Return the full sorted condition list for the given exp type.

    Conditions are sorted by descending cost (width×depth) so that
    round-robin shard assignment keeps per-shard runtime balanced.
    """
    if exp == "main":        raw = main_conditions()
    elif exp == "sigma":      raw = sigma_conditions()
    elif exp == "temp":       raw = temp_conditions()
    elif exp == "epoch":      raw = epoch_conditions()
    elif exp == "sigma_fine": raw = sigma_fine_conditions()
    elif exp == "temp_fine":  raw = temp_fine_conditions()
    elif exp == "phase":      raw = phase_conditions()
    elif exp == "epoch_fine": raw = epoch_fine_conditions()
    elif exp == "p6":
        raw = (sigma_fine_conditions() + temp_fine_conditions()
               + phase_conditions())
    elif exp == "p4_boost":   raw = epoch_fine_conditions()
    elif exp == "p6_all":
        raw = (sigma_fine_conditions() + temp_fine_conditions()
               + phase_conditions() + epoch_fine_conditions())
    else:
        raw = (main_conditions() + sigma_conditions()
               + temp_conditions() + epoch_conditions())

    # Sort by descending cost (width * depth) then distribute round-robin
    # across shards.  Without sorting, the product() order clusters heavy
    # conditions into a few shards, causing up to 42% runtime imbalance.
    # After sorting the imbalance drops below 1%.
    return sorted(raw, key=lambda c: -(c["width"] * max(c["depth"], 1)))


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _key(c: Dict) -> Tuple:
    return (c.get("exp_type"), c.get("dataset"), c.get("arch"),
            c.get("activation"), c.get("width"), c.get("depth"),
            c.get("seed"), c.get("sigma"), c.get("temperature"))


def load_done(path: str) -> Set[Tuple]:
    """Read completed condition keys from an existing shard file.

    Only records with kappa_input are considered done; error records
    are excluded so they re-run on the next invocation.
    """
    done: Set[Tuple] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "kappa_input" in r:
                    done.add(_key(r))
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model:             nn.Module,
    train_loader,
    eval_loader,
    device:            torch.device,
    n_epochs:          int,
    epoch_checkpoints: Optional[List[int]],
    X_raw:             torch.Tensor,
    X_flat:            torch.Tensor,
    Y:                 torch.Tensor,
    sigma:             float,
    n_classes:         int = 10,
) -> Dict:
    """
    Train for n_epochs with AdamW and cosine annealing.

    At each epoch in epoch_checkpoints, kappa_input, kappa_task, train_loss,
    and test_acc are all recorded.  Having accuracy alongside kappa at every
    checkpoint is what makes the kappa-vs-performance correlation analysis
    (Paper 5) possible — without per-epoch accuracy the correlation can only
    be computed at training end, not as a trajectory.

    Returns {train_loss, epoch_kappas}.
    """
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()

    epoch_kappas: List[Dict] = []
    last_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss   = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_loss += loss.item() * xb.shape[0]
            n          += xb.shape[0]
        sched.step()
        last_loss = total_loss / n

        if epoch_checkpoints and epoch in epoch_checkpoints:
            model.eval()

            # kappa measurement at this checkpoint
            with torch.no_grad():
                taps: Dict = {}
                model(X_raw[:KAPPA_CFG.n_eval].to(device), taps=taps)
            Z = taps.get("ffn_out")

            # accuracy at this checkpoint
            correct, total_n = 0, 0
            with torch.no_grad():
                for xb, yb in eval_loader:
                    xb, yb  = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total_n += yb.shape[0]
            acc_ep = round(correct / total_n if total_n > 0 else 0.0, 6)

            if Z is not None:
                try:
                    cfg_e = KappaCfg(
                        sigma=sigma,
                        run_sanity=False,
                        mi_train=KAPPA_CFG.mi_train,
                        est_steps=KAPPA_CFG.est_steps,
                        est_hidden=KAPPA_CFG.est_hidden,
                    )
                    r = measure_kappa(
                        X_flat[:KAPPA_CFG.n_eval], Z,
                        Y[:KAPPA_CFG.n_eval], cfg_e, device,
                        n_classes=n_classes,
                    )
                    epoch_kappas.append({
                        "epoch":       epoch,
                        "kappa_input": r.kappa_input,
                        "kappa_task":  r.kappa_task,
                        "train_loss":  round(last_loss, 6),
                        "test_acc":    acc_ep,
                    })
                except Exception:
                    pass

    return {"train_loss": round(last_loss, 6), "epoch_kappas": epoch_kappas}


def eval_accuracy(model: nn.Module, eval_loader, device: torch.device) -> float:
    """Compute top-1 accuracy over the full eval loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb  = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.shape[0]
    return round(correct / total if total > 0 else 0.0, 6)


# ---------------------------------------------------------------------------
# Single condition
# ---------------------------------------------------------------------------

def run_one(cond: Dict, device: torch.device, data_root: str) -> Dict:
    """Execute a single experimental condition end-to-end.

    Trains the model, measures canonical kappa (ffn_out tap),
    layerwise kappa, d_effective, and epoch checkpoints.
    Returns a flat dict ready for JSONL serialisation.
    """
    arch        = cond["arch"]
    width       = cond["width"]
    depth       = cond["depth"]
    activation  = cond["activation"]
    seed        = cond["seed"]
    dataset     = cond["dataset"]
    exp_type    = cond["exp_type"]
    sigma       = cond.get("sigma",       SIGMA_CANONICAL)
    temperature = cond.get("temperature", TEMP_CANONICAL)

    set_global_seed(seed)

    n_cls = get_n_classes(dataset)
    train_loader, eval_loader = get_loaders(
        dataset, data_root, batch_size=BATCH_SIZE, n_eval=KAPPA_CFG.n_eval,
    )
    X_raw, X_flat, Y = collect_eval(eval_loader, n=KAPPA_CFG.n_eval)

    model = build_model(
        arch, width, depth, activation,
        n_classes=n_cls, seed=seed,
        temperature=temperature,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    from core.config import THERMO_TRAIN_EPOCHS
    if exp_type == "epoch_fine":
        n_epochs_run      = THERMO_TRAIN_EPOCHS
        epoch_checkpoints = EPOCH_CHECKPOINTS
    elif exp_type == "epoch":
        n_epochs_run      = TRAIN_EPOCHS
        epoch_checkpoints = EPOCH_CHECKPOINTS
    else:
        n_epochs_run      = TRAIN_EPOCHS
        epoch_checkpoints = None

    t0 = time.time()
    train_result = train_model(
        model, train_loader, eval_loader, device,
        n_epochs=n_epochs_run,
        epoch_checkpoints=epoch_checkpoints,
        X_raw=X_raw, X_flat=X_flat, Y=Y,
        sigma=sigma,
        n_classes=n_cls,
    )
    test_acc   = eval_accuracy(model, eval_loader, device)
    train_time = round(time.time() - t0, 1)

    model.eval()
    with torch.no_grad():
        taps: Dict = {}
        if arch == "boltzmann":
            model(X_raw[:KAPPA_CFG.n_eval].to(device),
                  taps=taps, temperature=temperature)
        elif arch in ("transformer_preln", "transformer_postln"):
            # Pass temperature to attention softmax for T sweep.
            # At T=1.0 (TEMP_CANONICAL) this is identical to
            # standard scaled dot-product attention.
            model(X_raw[:KAPPA_CFG.n_eval].to(device),
                  taps=taps, temperature=temperature)
        else:
            model(X_raw[:KAPPA_CFG.n_eval].to(device), taps=taps)

    cfg = KappaCfg(
        sigma           = sigma,
        mi_train        = KAPPA_CFG.mi_train,
        mi_test         = KAPPA_CFG.mi_test,
        mi_seed         = KAPPA_CFG.mi_seed,
        noise_seed      = KAPPA_CFG.noise_seed,
        n_eval          = KAPPA_CFG.n_eval,
        est_hidden      = KAPPA_CFG.est_hidden,
        est_steps       = KAPPA_CFG.est_steps,
        est_lr          = KAPPA_CFG.est_lr,
        est_temperature = KAPPA_CFG.est_temperature,
        run_sanity      = KAPPA_CFG.run_sanity,
    )

    Z_final = taps.get("ffn_out")
    canon   = None
    if Z_final is not None:
        canon = measure_kappa(
            X_flat[:KAPPA_CFG.n_eval], Z_final,
            Y[:KAPPA_CFG.n_eval], cfg, device,
            n_classes=n_cls,
        )

    cfg_lw = KappaCfg(
        sigma      = sigma,
        run_sanity = False,
        mi_train   = min(KAPPA_CFG.mi_train, 2048),
        est_steps  = max(250, KAPPA_CFG.est_steps // 2),
        est_hidden = KAPPA_CFG.est_hidden,
    )
    lw    = measure_layerwise(taps, X_flat[:KAPPA_CFG.n_eval],
                               Y[:KAPPA_CFG.n_eval], cfg_lw, device,
                               n_classes=n_cls)
    d_eff = compute_d_effective(lw)

    c = canon
    return {
        "exp_type":    exp_type,
        "dataset":     dataset,
        "arch":        arch,
        "activation":  activation,
        "width":       width,
        "depth":       depth,
        "seed":        seed,
        "sigma":       sigma,
        "temperature": temperature,
        "n_params":    n_params,
        "kappa_input":       c.kappa_input       if c else None,
        "kappa_task":        c.kappa_task        if c else None,
        "mi_input_raw":      c.mi_input_raw      if c else None,
        "mi_task_raw":       c.mi_task_raw       if c else None,
        "d_z":               c.d_z               if c else None,
        "rms_z":             c.rms_z             if c else None,
        "saturated":         c.saturated         if c else None,
        "saturation_margin": c.saturation_margin if c else None,
        "log_batch":         c.log_batch         if c else None,
        "sanity_passed":     c.sanity_passed     if c else None,
        "permuted_kappa":    c.permuted_kappa    if c else None,
        "train_loss":        train_result["train_loss"],
        "test_acc":          test_acc,
        "train_time_s":      train_time,
        "layerwise": [
            {"tap_name":    r.tap_name,
             "kappa_input": r.kappa_input,
             "kappa_task":  r.kappa_task,
             "mi_input":    r.mi_input,
             "mi_task":     r.mi_task,
             "d_z":         r.d_z,
             "saturated":   r.saturated,
             "error":       r.error}
            for r in lw
        ],
        "d_effective":  d_eff,
        "epoch_kappas": train_result["epoch_kappas"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args, slice this shard's conditions, and run them.

    Appends each result to the shard JSONL file immediately so that
    a crash loses at most one condition rather than the whole shard.
    """
    p = argparse.ArgumentParser(description="Icon_Empirical runner")
    p.add_argument("--gpu",       type=int, default=0)
    p.add_argument("--shard",     type=int, default=0)
    p.add_argument("--n_shards",  type=int, default=N_SHARDS)
    p.add_argument("--out",       type=str, default="results/shard_{shard}.jsonl")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--exp",       type=str, default="all",
                   choices=["all", "main", "sigma", "temp", "epoch", "sigma_fine", "temp_fine", "phase", "epoch_fine", "p6", "p4_boost", "p6_all"])
    args = p.parse_args()

    out_path = args.out.format(shard=args.shard)
    device   = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    print(f"[Icon_Empirical] shard={args.shard}/{args.n_shards}  device={device}  exp={args.exp}")
    print(f"[Icon_Empirical] output -> {out_path}")

    all_conds   = get_conditions(args.exp)
    shard_conds = [c for i, c in enumerate(all_conds)
                   if i % args.n_shards == args.shard]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done      = load_done(out_path)
    remaining = [c for c in shard_conds if _key(c) not in done]

    print(f"[Icon_Empirical] total={len(all_conds)}  shard={len(shard_conds)}  "
          f"done={len(done)}  remaining={len(remaining)}")

    with open(out_path, "a") as f:
        for i, cond in enumerate(remaining):
            print(f"\n[{i+1}/{len(remaining)}] {cond}", flush=True)
            t0 = time.time()
            try:
                record = run_one(cond, device, args.data_root)
            except Exception as e:
                record = {**cond, "error": str(e),
                          "traceback": traceback.format_exc()}
            record["wall_time_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  kappa_input={record.get('kappa_input')}  "
                  f"kappa_task={record.get('kappa_task')}  "
                  f"acc={record.get('test_acc')}  "
                  f"{record['wall_time_s']}s")

    print(f"\n[Icon_Empirical] shard {args.shard} done.")


if __name__ == "__main__":
    main()

