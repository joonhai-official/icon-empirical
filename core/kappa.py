# core/kappa.py
#
# Core kappa measurement: kappa_input and kappa_task.
#
# Definitions
# -----------
#   kappa_input = I(X; Z_tilde(sigma)) / d_z    input information density
#   kappa_task  = I(Y; Z_tilde(sigma)) / d_z    task information density
#   Z_tilde = Z + sigma * RMS(Z) * epsilon,   epsilon ~ N(0, I)
#
# Both estimators share the same Z_tilde so the comparison is fair.
# Separate critic seeds (mi_seed, mi_seed+1) prevent the two networks from
# co-adapting and producing correlated estimates.
#
# Saturation check
# ----------------
# A tap is marked saturated when:
#   (log(batch_size) - MI_raw) / log(batch_size) < SAT_THRESH
# i.e. MI > 0.95 * log(batch_size).  Evaluated in MI space (nats)
# so the verdict is independent of d_z.
#
# Sanity (permutation) check
# --------------------------
# Permuting Z breaks all statistical dependence with X.
# The resulting permuted_kappa = permuted_MI / d_z must stay below
# SANITY_THRESH (0.1).  Comparing raw permuted_MI to 0.1 always fails
# because MI is in nats (~2-6); the d_z division brings it to kappa scale.
#
# d_effective
# -----------
#   d_eff(l) = I(X; h_0) - I(X; h_l)    [nats]
# h_0 is the first layer output (closest to input).
# By the Data Processing Inequality I(X; h_l) can only decrease as l
# increases, so d_eff is non-negative and monotonically non-decreasing.
# The fixed point d* is the layer where the increment drops below D_EFF_EPS.

import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .noise_channel import NoiseChannel
from .mi_estimator  import build_estimator
from .config        import KappaCfg, SAT_THRESH, SANITY_THRESH, D_EFF_EPS, INFONCE_BATCH


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class KappaResult:
    """Return type of measure_kappa().

    kappa_input = I(X;Z̃)/d_z — input information density.
    kappa_task  = I(Y;Z̃)/d_z — task information density.
    saturated   = True when MI > 0.95·log(batch_size).
    sanity_passed = True when permuted_kappa < 0.1.
    """
    kappa_input:       float
    kappa_task:        float
    mi_input_raw:      float    # raw nats from InfoNCE
    mi_task_raw:       float
    d_z:               int
    sigma:             float
    rms_z:             float
    saturated:         bool
    saturation_margin: float    # (log_batch - MI) / log_batch
    log_batch:         float    # log(min(512, n_train))
    sanity_passed:     bool
    permuted_kappa:    float    # permuted_MI / d_z
    # Note: per-condition d_effective is computed separately via
    # compute_d_effective(layerwise_results) and stored in the runner record.


@dataclass
class LayerwiseResult:
    """Per-tap result from measure_layerwise().

    One instance per tap (L0, L1, ..., ffn_out).
    error is None on success; non-None records the exception message.
    """
    tap_name:    str
    kappa_input: float
    kappa_task:  float
    mi_input:    float
    mi_task:     float
    d_z:         int
    saturated:   bool
    error:       Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pool(z: torch.Tensor) -> torch.Tensor:
    """
    Collapse spatial / sequence dimensions so z is always [N, d].
      [N, d]       -> identity
      [N, T, d]    -> mean over T   (transformer / RNN sequence)
      [N, C, H, W] -> mean over H,W (CNN spatial map)
    """
    if z.dim() == 2:
        return z
    elif z.dim() == 3:
        return z.mean(dim=1)
    elif z.dim() == 4:
        return z.mean(dim=[2, 3])
    raise ValueError(f"pool: unsupported shape {tuple(z.shape)}")


def encode_y(
    y:         torch.Tensor,
    device:    torch.device,
    n_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert labels to a 2-D float tensor for InfoNCE.
    Integer class labels  -> one-hot vectors of length n_classes.
    Float tensors (regression targets) -> passed through as-is.

    n_classes must be provided for datasets whose classes are not all
    represented in the eval subset (e.g. TinyImageNet with 200 classes
    but only 4096 eval samples drawn from the front of the sorted val set,
    which covers only ~82 classes).  Without n_classes the one-hot dimension
    is inferred as max(Y)+1, which underestimates the true class count and
    makes kappa_task not directly comparable across datasets.
    """
    y = y.to(device)
    if y.dtype in (torch.long, torch.int32, torch.int64):
        n_cls = n_classes if n_classes is not None else int(y.max().item()) + 1
        oh = torch.zeros(y.shape[0], n_cls, device=device)
        oh.scatter_(1, y.view(-1, 1).long(), 1.0)
        return oh
    return y.float()


# ---------------------------------------------------------------------------
# Single-tap measurement
# ---------------------------------------------------------------------------

def measure_kappa(
    X:         torch.Tensor,    # [N, d_x] flat input, already pooled
    Z:         torch.Tensor,    # [N, ...] layer output, auto-pooled here
    Y:         torch.Tensor,    # [N] class labels or [N, d_y] targets
    cfg:       KappaCfg,
    device:    torch.device,
    n_classes: Optional[int] = None,  # total class count; pass get_n_classes(dataset)
) -> KappaResult:
    """
    Measure kappa_input and kappa_task at a single tap point.

    Both share Z_tilde (same noise seed) so the comparison is fair.
    The critic hidden dimension is est_hidden (default 128).  When d_x is
    large (e.g. 3072 for flat CIFAR), the critic acts as a lossy compressor
    of X — this lowers absolute kappa_input values but does not affect
    the relative comparisons that drive the Width Law and C_arch analyses,
    because every condition uses an identical critic architecture.
    """
    X  = pool(X).to(device)
    Z  = pool(Z).to(device)
    d_z = Z.shape[1]
    N   = X.shape[0]

    nc      = NoiseChannel(sigma=cfg.sigma, seed=cfg.noise_seed)
    Z_tilde = nc(Z)
    rms_z   = float(nc.rms(Z).item())

    n_tr = min(cfg.mi_train, N)
    X_tr = X[:n_tr]
    Z_tr = Z_tilde[:n_tr]
    Y_2d = encode_y(Y[:n_tr], device, n_classes=n_classes)

    # kappa_input: I(X; Z_tilde) / d_z
    torch.manual_seed(cfg.mi_seed)
    np.random.seed(cfg.mi_seed)
    est_inp = build_estimator(
        X.shape[1], d_z, device,
        hidden=cfg.est_hidden, steps=cfg.est_steps,
        lr=cfg.est_lr, temperature=cfg.est_temperature,
        seed=cfg.mi_seed,
    )
    mi_input    = est_inp.estimate(X_tr, Z_tr)
    kappa_input = mi_input / d_z

    # kappa_task: I(Y; Z_tilde) / d_z
    # seed offset by 1 so the two critics start from different initialisations
    torch.manual_seed(cfg.mi_seed + 1)
    np.random.seed(cfg.mi_seed + 1)
    est_tsk = build_estimator(
        Y_2d.shape[1], d_z, device,
        hidden=cfg.est_hidden, steps=cfg.est_steps,
        lr=cfg.est_lr, temperature=cfg.est_temperature,
        seed=cfg.mi_seed + 1,
    )
    mi_task    = est_tsk.estimate(Y_2d, Z_tr)
    kappa_task = mi_task / d_z

    # saturation: in MI space so the verdict is independent of d_z
    log_batch  = math.log(min(INFONCE_BATCH, n_tr))
    sat_margin = (log_batch - mi_input) / log_batch if log_batch > 0 else 0.0
    saturated  = sat_margin < SAT_THRESH

    # permutation sanity check
    sanity_passed  = True
    permuted_kappa = 0.0

    if cfg.run_sanity and N > n_tr:
        n_te   = min(cfg.mi_test, N - n_tr)
        X_te   = X[n_tr: n_tr + n_te]
        Z_te   = Z_tilde[n_tr: n_tr + n_te]
        perm   = torch.randperm(n_te, device=device)
        nc2    = NoiseChannel(sigma=cfg.sigma, seed=cfg.noise_seed + 99999)
        Z_perm = nc2(Z_te[perm])

        torch.manual_seed(cfg.mi_seed + 2)
        est_sn = build_estimator(
            X.shape[1], d_z, device,
            hidden=cfg.est_hidden, steps=cfg.est_steps,
            lr=cfg.est_lr, temperature=cfg.est_temperature,
            seed=cfg.mi_seed + 2,
        )
        perm_mi_raw    = est_sn.estimate(X_te, Z_perm)
        permuted_kappa = perm_mi_raw / d_z
        sanity_passed  = permuted_kappa < SANITY_THRESH

    return KappaResult(
        kappa_input       = round(kappa_input, 8),
        kappa_task        = round(kappa_task,  8),
        mi_input_raw      = round(mi_input,    6),
        mi_task_raw       = round(mi_task,     6),
        d_z               = d_z,
        sigma             = cfg.sigma,
        rms_z             = round(rms_z,       6),
        saturated         = saturated,
        saturation_margin = round(sat_margin,  6),
        log_batch         = round(log_batch,   6),
        sanity_passed     = sanity_passed,
        permuted_kappa    = round(permuted_kappa, 8),
    )


# ---------------------------------------------------------------------------
# Layerwise measurement
# ---------------------------------------------------------------------------

def _tap_sort_key(name: str) -> Tuple:
    """
    Sort order: L0, L1, ..., Ln  (layer taps)
              then B0, B1, ...    (branch taps for parallel arch)
              then ffn_out        (canonical tap)
              then anything else
    """
    if name.startswith("L"):
        try:
            return (0, int(name.split("_")[0][1:]))
        except ValueError:
            pass
    if name.startswith("B"):
        try:
            return (1, int(name.split("_")[0][1:]))
        except ValueError:
            pass
    if name == "ffn_out":
        return (2, 0)
    return (3, 0)


def measure_layerwise(
    taps:      Dict[str, torch.Tensor],
    X:         torch.Tensor,
    Y:         torch.Tensor,
    cfg:       KappaCfg,
    device:    torch.device,
    n_classes: Optional[int] = None,
) -> List[LayerwiseResult]:
    """
    Run measure_kappa on every tap except 'input', in forward-pass order.
    Per-tap errors are caught and recorded; they do not abort the loop.
    """
    X_p = pool(X).to(device)

    sorted_taps = sorted(
        [(k, v) for k, v in taps.items() if k != "input"],
        key=lambda kv: _tap_sort_key(kv[0]),
    )

    results: List[LayerwiseResult] = []
    for tap_name, Z in sorted_taps:
        if not isinstance(Z, torch.Tensor):
            continue
        try:
            r = measure_kappa(X_p, Z, Y, cfg, device, n_classes=n_classes)
            results.append(LayerwiseResult(
                tap_name    = tap_name,
                kappa_input = r.kappa_input,
                kappa_task  = r.kappa_task,
                mi_input    = r.mi_input_raw,
                mi_task     = r.mi_task_raw,
                d_z         = r.d_z,
                saturated   = r.saturated,
            ))
        except Exception as exc:
            results.append(LayerwiseResult(
                tap_name    = tap_name,
                kappa_input = 0.0,
                kappa_task  = 0.0,
                mi_input    = 0.0,
                mi_task     = 0.0,
                d_z         = 0,
                saturated   = False,
                error       = str(exc),
            ))

    return results


# ---------------------------------------------------------------------------
# d_effective computation
# ---------------------------------------------------------------------------

def compute_d_effective(layerwise: List[LayerwiseResult]) -> List[Dict]:
    """
    d_eff(l) = I(X; h_0) - I(X; h_l)   for each tap l.

    h_0 is the first valid L-tap (closest to the input).
    d_eff is always >= 0 by the Data Processing Inequality.
    When delta_d_eff drops below D_EFF_EPS the layer is declared the
    fixed point d*.
    """
    mi_h0: Optional[float] = None
    for r in layerwise:
        if r.error is None and r.tap_name.startswith("L"):
            mi_h0 = r.mi_input
            break

    if mi_h0 is None:
        return []

    records    = []
    prev_d_eff = 0.0

    for r in layerwise:
        if r.error is not None:
            continue
        # DPI guarantees d_eff >= 0 in theory.  InfoNCE estimation noise
        # can produce small negative values; clamp to enforce the constraint.
        d_eff = max(0.0, mi_h0 - r.mi_input)
        delta = d_eff - prev_d_eff
        is_fp = abs(delta) < D_EFF_EPS

        records.append({
            "tap_name":       r.tap_name,
            "d_effective":    round(d_eff, 6),
            "delta_d_eff":    round(delta, 6),
            "is_fixed_point": is_fp,
        })
        prev_d_eff = d_eff

    return records
