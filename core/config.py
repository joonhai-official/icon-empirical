# core/config.py
#
# Central configuration for all Icon_Empirical experiments.
# Every hyperparameter lives here — nothing hardcoded in experiment code.
# Changing a value here propagates everywhere automatically.
#
# Layout
# ------
# 1. Experiment grid   : what conditions to sweep
# 2. Training          : optimizer / scheduler settings
# 3. Kappa measurement : MI estimator config
# 4. Pilot             : fast smoke-test subset
# 5. Formula constants : thresholds tied to published math
# 6. Reproducibility   : global seed setter

from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# 1. Experiment grid
# ---------------------------------------------------------------------------

# Primary dataset: CIFAR-10. CIFAR-100 and TinyImageNet supported but not used in this series.
DATASETS: List[str] = ["cifar10"]

# Each architecture must implement forward(x, taps=None) -> logits.
ARCHS: List[str] = [
    "mlp",
    "cnn",
    "gru",
    "transformer_preln",
    "transformer_postln",
    "serial",
    "parallel",
    "boltzmann",
]

# Activation functions applied uniformly so topology and activation effects
# can be separated in the C_arch analysis.
ACTIVATIONS: List[str] = ["relu", "gelu", "tanh"]

# 16 and 1024 anchor the extremes; they test whether the Width Law holds
# beyond the comfortable 64-512 range.
WIDTHS: List[int] = [16, 32, 64, 128, 256, 512]

# Log-spaced depths.  Linear spacing would waste compute past the Transformer
# fixed point around d=8.
DEPTHS: List[int] = [1, 2, 4, 8, 16]

# Three seeds surface instability without tripling total compute.
SEEDS: List[int] = [0, 1, 2]

# --- sigma sweep ---
# canonical sigma used for all main-experiment kappa measurements.
# The sweep finds sigma* = argmax_sigma kappa(sigma), testing whether
# each architecture has an intrinsic noise scale.
SIGMA_CANONICAL:   float       = 0.1
SIGMAS_SWEEP:      List[float] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SIGMA_SWEEP_WIDTH: int         = 256   # held constant so only sigma varies
SIGMA_SWEEP_DEPTH: int         = 4

# --- T sweep (Boltzmann + Transformer) ---
# Temperature T controls:
#   Boltzmann    : sigmoid sharpness in mean-field update
#   Transformer  : attention softmax temperature (standard: T=1)
# Range spans 4 orders of magnitude to capture any phase transition.
# Boltzmann: full width/depth grid.  Transformer: representative subset
# (w=[64,256], d=4).  Together they cost ~9h extra vs Boltzmann-only.
TEMP_CANONICAL:    float       = 1.0
TEMPS_SWEEP:       List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
TEMP_SWEEP_WIDTHS: List[int]   = [64, 256, 512]
TEMP_SWEEP_DEPTHS: List[int]   = [1, 4, 8, 16, 32]
# Architectures in T sweep.
#   boltzmann        : sigmoid sharpness → expected T-sensitive
#   transformer_preln: attention softmax → expected T-invariant
# Comparing these two directly is the core P3 claim.
TEMP_SWEEP_ARCHS:  List[str]   = ["boltzmann", "transformer_preln"]

# --- kappa_epoch ---
# kappa measured at fixed training checkpoints for learning-dynamics analysis.
# transformer_preln only — it shows the clearest trajectory signal.
EPOCH_ARCHS:       List[str] = ["transformer_preln"]
EPOCH_WIDTHS:      List[int] = [64, 256]
EPOCH_DEPTHS:      List[int] = [4]
EPOCH_CHECKPOINTS: List[int] = [1, 2, 5, 10, 20, 50, 100, 200]


# ---------------------------------------------------------------------------
# 2. Training settings
# ---------------------------------------------------------------------------

TRAIN_EPOCHS: int   = 10
BATCH_SIZE:   int   = 256
LR:           float = 1e-3
WEIGHT_DECAY: float = 1e-4
GRAD_CLIP:    float = 1.0    # gradient norm clip; prevents RNN blow-up


# ---------------------------------------------------------------------------
# 3. Kappa measurement config
# ---------------------------------------------------------------------------

@dataclass
class KappaCfg:
    """Hyperparameters for a single kappa measurement.

    Passed to measure_kappa() and measure_layerwise().
    KAPPA_CFG is the production instance; KAPPA_CFG_PILOT is
    a lightweight version for the smoke-test pilot.
    """
    sigma:           float = SIGMA_CANONICAL
    mi_train:        int   = 4096   # samples fed to the MI estimator
    mi_test:         int   = 4096   # held-out samples for the permutation check
    mi_seed:         int   = 0      # seeds critic init and data shuffling
    noise_seed:      int   = 42     # seeds the noise channel so Z_tilde is reproducible
    n_eval:          int   = 8192   # total eval samples collected before measurement
    est_hidden:      int   = 128    # critic MLP hidden size
    est_steps:       int   = 500    # gradient steps for the InfoNCE critic
    est_lr:          float = 1e-3
    est_temperature: float = 0.1    # InfoNCE softmax temperature
    run_sanity:      bool  = True   # whether to run the permutation sanity check


# Production config.
KAPPA_CFG = KappaCfg()

# Lightweight config for pilot and epoch checkpoints — roughly 10x faster.
KAPPA_CFG_PILOT = KappaCfg(
    mi_train=512,
    mi_test=512,
    n_eval=1024,
    est_steps=200,
    run_sanity=False,
)


# ---------------------------------------------------------------------------
# 4. Pilot settings
# ---------------------------------------------------------------------------

PILOT_ARCHS:   List[str] = ["mlp", "transformer_preln"]
PILOT_WIDTHS:  List[int] = [64, 256]
PILOT_DEPTHS:  List[int] = [1, 4]
PILOT_SEEDS:   List[int] = [0]
PILOT_DATASET: str       = "cifar10"
PILOT_EPOCHS:  int       = 5


# ---------------------------------------------------------------------------
# 5. Formula constants
# ---------------------------------------------------------------------------

# Width Law R-squared threshold below which we do not claim the law holds.
# Note on d_z: kappa = MI / d_z where d_z is the ffn_out dimension.
# All architectures produce d_z = width for their ffn_out tap.
# CNN uses a Linear(width*4, width) projection after GAP so that
# its d_z = width matches all other architectures, enabling direct
# C_arch comparison on the same scale.
ALPHA_R2_MIN: float = 0.99

# Theoretical exponent from kappa = C * w^alpha.
# alpha = -1 follows from kappa = I/d_z and the information saturation argument.
ALPHA_THEORY: float = -1.0

# Saturation margin threshold.
# A tap is flagged saturated when:
#   (log(batch_size) - MI_raw) / log(batch_size) < SAT_THRESH
# Evaluated in MI space (nats) so the verdict is independent of d_z.
SAT_THRESH: float = 0.05

# d_effective fixed-point epsilon.
# Layer l is a fixed point when |delta_d_eff(l)| < D_EFF_EPS nats.
D_EFF_EPS: float = 0.01

# InfoNCE batch size.  The MI estimator subsamples this many pairs per step.
# Larger batches give tighter lower bounds but increase per-step cost.
# 512 is the practical sweet spot for our hardware and step budget.
# kappa_max = log(INFONCE_BATCH) / d_z is the theoretical InfoNCE ceiling.
INFONCE_BATCH: int = 512

# Sanity check threshold.
# permuted_kappa = permuted_MI / d_z must stay below SANITY_THRESH.
# Comparing raw permuted_MI to 0.1 always fails because MI is in nats (~2-6);
# dividing by d_z first brings the quantity to the kappa scale (~0.001-0.1).
SANITY_THRESH: float = 0.1

N_SHARDS: int = 8   # number of parallel GPU workers


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """
    Lock all randomness sources before building a model or loading data.
    Called at the start of every run_one() invocation.

    cudnn.deterministic=True disables non-deterministic CUDA kernels at a
    small speed cost (~5-10%), which is worth it for bitwise reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# 7. P6 Thermo — extended physics experiments
# ---------------------------------------------------------------------------

# Fine-grained sigma sweep: 12 points spanning 4 orders of magnitude.
# Resolves the bell-curve shape around sigma* more precisely.
SIGMAS_FINE: List[float] = [
    0.001, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
]

# Fine-grained T sweep: 7 points spanning 6 orders of magnitude.
# Tests whether Boltzmann becomes T-sensitive with longer training.
TEMPS_FINE: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
THERMO_TRAIN_EPOCHS: int = 100   # longer training for T re-experiment

# Fine-grained depth sweep for phase transition search.
# Serial and Transformer Post-LN are depth-sensitive — scanning
# d=1..16 at unit steps locates the critical depth d_c where kappa collapses.
DEPTHS_FINE: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
PHASE_ARCHS: List[str] = ["serial", "transformer_postln"]

# ---------------------------------------------------------------------------
# 8. P7 Theory — no new experiments, reuses full.jsonl
# ---------------------------------------------------------------------------

# P8 HW — parameter count and FLOPs measured analytically from model objects.
