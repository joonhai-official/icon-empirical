# Icon_Empirical

**Reproducibility codebase for *Information Flow in Neural Networks: An Empirical Study* (Park, 2026).**

This repository contains the complete experimental code and raw results used in the paper. Anyone can clone, run the analysis scripts, and reproduce every numerical claim in the paper from the raw JSONL files.

> **Looking for the framework?** This repository is the *empirical* codebase tied to a specific study (8 architectures on CIFAR-10, fixed batch size B=512). The general-purpose measurement framework that this work motivates — **Icon** — is being developed as a separate library that can be applied to arbitrary representation-bearing systems. See Section 12 of the paper for the framework specification (in preparation by the present author).

---

## Quick Reproduction

```bash
# Clone, install, and reproduce every number in the paper:
git clone https://github.com/joonhai-official/Icon_Empirical
cd Icon_Empirical
pip install -r requirements.txt
python papers/run_all.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

Each analysis script reads the released raw JSONL data and prints results that match the paper's tables and numbers.

---

## Paper-to-Code Mapping

This codebase maps directly onto the paper's structure. Use this table to find the code behind any claim in the paper:

| Paper Section | Topic | Analysis Script |
|---|---|---|
| §3 | Activation function effects | `papers/p1_width_law.py` (activation slice) |
| §4 | Width Scaling Law (κ ∝ 1/w) | `papers/p1_width_law.py` |
| §5 | Architecture-Specific Constants C_arch | `papers/p1_width_law.py` (C_arch table) |
| §6 | Two-Variable Scaling κ(w, d) | `papers/p4_unified.py` |
| §7 | Phase Transitions, critical depth d_c | `papers/p2_depth_law.py`, `papers/p6_thermo.py` |
| §8 | Training Dynamics | `papers/p5_dynamics.py` |
| §9 | Information Bottleneck Ratio η_t | `papers/p6_thermo.py` |
| §10 | Inverse Design | `papers/p8_hw.py` |
| §11 | Layer-wise Patterns | `papers/p2_depth_law.py` (layerwise section) |
| Appendix B | σ-sweep (robustness) | `papers/p3_physics.py` (σ section) |
| Appendix C | Temperature sweep | `papers/p3_physics.py` (T section) |
| App. F, §2 | Theory connections (IB, scaling) | `papers/p7_theory.py` |

Run any individual analysis to see its outputs:

```bash
python papers/p1_width_law.py --data results/full.jsonl
python papers/p2_depth_law.py --data results/full.jsonl
python papers/p3_physics.py   --data results/full.jsonl
python papers/p4_unified.py   --data results/full.jsonl
python papers/p5_dynamics.py  --data results/full.jsonl
python papers/p6_thermo.py    --data results/full.jsonl --data_p6 results_p6/full.jsonl
python papers/p7_theory.py    --data results/full.jsonl
python papers/p8_hw.py        --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

---

## Core Metric (Paper §2.1)

```
κ_input = I(X ; Z̃(σ)) / d_z       input information density
κ_task  = I(Y ; Z̃(σ)) / d_z       task information density

Z̃ = Z + σ · RMS(Z) · ε,   ε ~ N(0, I),   σ = 0.1
```

Mutual information is estimated via the InfoNCE lower bound with batch size B = 512. All eight architectures expose a unified `ffn_out` tap of shape `[B, width]` (d_z = width). CNN uses a GAP → Linear(width×4 → width) projection to match d_z.

---

## Sanity Filter (Paper §2.4–§2.5)

Records are filtered before analysis using the paper's **Φ_default** filter:

```
Φ_default = F_main ∧ F_sanity ∧ F_ki_pos

  F_main    = exp_type == "main"
  F_sanity  = sanity_passed                 (κ_permuted < 0.1)
  F_ki_pos  = kappa_input > 0
```

All `papers/p*.py` scripts apply this filter consistently. The resulting working set is **n = 633 records**, matching every figure and table in the paper.

---

## Data

| File | Records | Description | sha256 |
|---|---|---|---|
| `results/full.jsonl` | 1,405 | Main grid + σ/T/dynamics sweeps | `4f1f1959…2b3058a5` |
| `results_p6/full.jsonl` | 836 | Phase transition (w, d) grid | `3113f068…95d65d42` |

**Total: 2,241 records.** Both files are released as-is; running any script produces results identical to the paper's tables.

---

## Project Structure

```
Icon_Empirical/
├── papers/                     analysis scripts (one per paper section group)
│   ├── p1_width_law.py         §3 activation, §4 width law, §5 C_arch
│   ├── p2_depth_law.py         §7 depth classification, §11 layerwise
│   ├── p3_physics.py           App. B σ-sweep, App. C T-sweep
│   ├── p4_unified.py           §6 unified κ(w, d) scaling
│   ├── p5_dynamics.py          §8 training dynamics
│   ├── p6_thermo.py            §7 phase transitions, §9 IB ratio η_t
│   ├── p7_theory.py            §2, App. F — connections to existing theory
│   ├── p8_hw.py                §10 inverse design
│   └── run_all.py              run all eight scripts sequentially
│
├── core/                       shared measurement infrastructure
│   ├── config.py               all hyperparameters and sweep definitions
│   ├── kappa.py                κ measurement (κ_input, κ_task, layerwise)
│   ├── mi_estimator.py         InfoNCE mutual information estimator
│   ├── noise_channel.py        Z̃ = Z + σ·RMS(Z)·ε noise injection
│   └── dataset.py              CIFAR-10 / CIFAR-100 / TinyImageNet loaders
│
├── models/
│   └── __init__.py             8 architecture implementations
│
├── experiments/
│   ├── runner.py               experiment loop (shard / resume / JSONL)
│   └── aggregate.py            shard merging, sanity checks
│
├── tests/
│   ├── test_formulas.py        formula unit tests (47 assertions)
│   ├── test_models.py          forward shape, tap d_z, reproducibility
│   └── pilot.py                end-to-end smoke test
│
├── scripts/
│   ├── setup.sh                install → test → pilot
│   └── run.sh                  8-GPU parallel execution (p4d.24xlarge)
│
├── results/                    raw JSONL + pre-computed analysis outputs
│   ├── full.jsonl              1,405 measurement records (main + sweeps)
│   └── analysis_p[1-8].json    pre-computed outputs of each analysis script
│
├── results_p6/                 raw JSONL for the phase-transition grid
│   ├── full.jsonl              836 measurement records
│   └── analysis_p6.json        pre-computed output of p6_thermo.py
│
└── requirements.txt
```

---

## Architectures

| Key | Design | Notes |
|---|---|---|
| `mlp` | fully connected, residual blocks | |
| `cnn` | 3-stage ResNet-V1, stride-2 downsampling | d_z projection applied; w=512 OOM |
| `gru` | 4×4 patch sequence, mean-pool | depth capped at 8; d > 8 identical to d=8 |
| `transformer_preln` | Pre-LN transformer, learnable PE | most depth-stable architecture |
| `transformer_postln` | Post-LN transformer (BERT-style) | depth-sensitive; κ_task collapses at d ≥ 8 |
| `serial` | feedforward stack, no residual connections | hard collapse at d_c(w); d_c decreases with width |
| `parallel` | parallel branches with residual merge | depth-invariant |
| `boltzmann` | mean-field RBM, backprop-trained | exact depth invariance: CV = 0% across d = 1–16 |

---

## Experiment Grid

The full study uses two grids:

**Main grid** (`results/full.jsonl`, 1,405 records):

| Variable | Values |
|---|---|
| Architecture | 8 (mlp, cnn, gru, transformer_preln, transformer_postln, serial, parallel, boltzmann) |
| Width | 16, 32, 64, 128, 256, 512 |
| Depth | 1, 2, 4, 8, 16, 32 |
| Activation | relu, gelu, tanh |
| Seed | 0, 1, 2 |
| Dataset | CIFAR-10 |

Includes the main full-grid sweep plus σ-sweep, T-sweep, and per-epoch dynamics measurements.

**Phase-transition grid** (`results_p6/full.jsonl`, 836 records):

| Variable | Values |
|---|---|
| Architecture | serial, transformer_postln (collapse-sensitive subset) |
| Width | 16, 32, 64, 128, 256, 512 |
| Depth | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16 (fine-grained near d_c) |

Used for resolving phase-transition boundaries d_c in §7.

**Hardware:** AWS p4d.24xlarge (8 × A100 40 GB), 8 parallel shards, ~37 hours total wall-clock.

---

## Running from Scratch (Optional)

If you want to regenerate the JSONL files from raw experiments instead of using the released ones:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run formula + model tests
python -m pytest tests/

# 3. Smoke test (single config, ~2 min)
python tests/pilot.py

# 4. Full experiment run (requires 8 GPUs, ~37 hours)
bash scripts/run.sh all

# 5. Merge shards
python experiments/aggregate.py

# 6. Run all analyses (this matches the paper)
python papers/run_all.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

For most users, skipping steps 4–5 and running step 6 directly on the released JSONL is sufficient.

---

## Output Format

Each completed condition is written as one JSON line to `results/shard_N.jsonl`:

```json
{
  "exp_type": "main",
  "arch": "transformer_preln",
  "width": 256,
  "depth": 4,
  "activation": "relu",
  "seed": 0,
  "dataset": "cifar10",
  "sigma": 0.1,
  "temperature": 1.0,
  "kappa_input": 0.02354,
  "kappa_task": 0.00821,
  "d_z": 256,
  "saturated": false,
  "sanity_passed": true,
  "test_acc": 0.5832,
  "layerwise": [...],
  "epoch_kappas": [...]
}
```

Full schema in paper Appendix A.

---

## Reproducibility

```python
set_global_seed(seed)       # random, numpy, torch, cuda
KappaCfg.mi_seed    = 0     # critic initialization + data shuffle
KappaCfg.noise_seed = 42    # Z̃ noise channel
```

`torch.backends.cudnn.deterministic = True` is set globally. Each record is uniquely identified by `(arch, width, depth, activation, seed, sigma, dataset)`.

---

## Citation

```bibtex
@misc{park2026icon_empirical,
  author = {Park, JoonHa},
  title  = {Information Flow in Neural Networks: An Empirical Study},
  year   = {2026},
  note   = {arXiv preprint}
}
```
