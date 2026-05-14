# ICON Empirical

**Reproducibility package for _Information Capacity in Neural Networks: An Empirical Study of κ Scaling_ (Park, 2026).**

This repository contains the Phase 0 empirical codebase, raw JSONL measurement records, analysis scripts, and manuscript for the ICON research program.

The goal of this repository is simple:

> Make every numerical claim in the Phase 0 empirical paper reproducible from released data and code.

This repository is the empirical anchor for ICON. It is not the general-purpose framework library.

---

## Looking for the ICON framework?

The general-purpose framework is released separately:

- Framework repository: [`joonhai-official/icon`](https://github.com/joonhai-official/icon)

This repository, `icon-empirical`, contains the empirical reproduction package for one specific Phase 0 study:

- Dataset: CIFAR-10
- Architectures: 8 neural architecture families
- Measurement regime: InfoNCE-based κ measurements with fixed batch size `B = 512`
- Main output: raw JSONL records, analysis scripts, and paper-level numerical reproduction

---

## Paper

- [Information Capacity in Neural Networks: An Empirical Study of κ Scaling](manuscripts/Information_Capacity_in_Neural_Networks.pdf)

---

## Status

This is an early Phase 0 research release.

The paper reports four empirical observations plus one phenomenological scaling pattern:

1. **Phase-transition-like collapse** in residual-free architectures
2. **η_t = κ_task / κ_input** as a dimension-normalization-canceling information-bottleneck ratio
3. **Training-dynamics evidence** linking η_t and task performance during learning
4. **Forward model / inverse-design proof of concept** within the measured grid
5. **Phenomenological width scaling** under the fixed Phase 0 measurement regime

The width-scaling result is reported cautiously. It is not claimed as a universal law. The paper explicitly treats it as a phenomenological observation under the present InfoNCE finite-batch regime.

---

## Quick Reproduction

Most users do **not** need to rerun the full experiment grid.

To reproduce the paper's numerical claims from the released JSONL files:

```bash
git clone https://github.com/joonhai-official/icon-empirical
cd icon-empirical
pip install -r requirements.txt
python papers/run_all.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

Each analysis script reads the released raw JSONL data and prints results corresponding to the paper's reported tables, figures, and numerical claims.

---

## Paper-to-Code Mapping

This codebase maps directly onto the paper's structure.

| Paper Section | Topic | Analysis Script |
|---|---|---|
| §3 | Activation function effects | `papers/p1_width_law.py` |
| §4 | Phenomenological width scaling | `papers/p1_width_law.py` |
| §5 | Architecture-specific constants | `papers/p1_width_law.py` |
| §6 | Two-variable scaling κ(w, d) | `papers/p4_unified.py` |
| §7 | Phase transitions and critical depth | `papers/p2_depth_law.py`, `papers/p6_thermo.py` |
| §8 | Training dynamics | `papers/p5_dynamics.py` |
| §9 | Information bottleneck ratio η_t | `papers/p6_thermo.py` |
| §10 | Forward model and inverse-design proof of concept | `papers/p8_hw.py` |
| §11 | Layer-wise patterns | `papers/p2_depth_law.py` |
| Appendix B | σ-sweep robustness | `papers/p3_physics.py` |
| Appendix C | Temperature sweep | `papers/p3_physics.py` |
| Appendix F | Theory connections | `papers/p7_theory.py` |

Run all analyses:

```bash
python papers/run_all.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

Run individual analyses:

```bash
python papers/p1_width_law.py --data results/full.jsonl
python papers/p2_depth_law.py --data results/full.jsonl
python papers/p3_physics.py --data results/full.jsonl
python papers/p4_unified.py --data results/full.jsonl
python papers/p5_dynamics.py --data results/full.jsonl
python papers/p6_thermo.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
python papers/p7_theory.py --data results/full.jsonl
python papers/p8_hw.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

---

## Core Measurement

The main quantity is information capacity per latent dimension:

```text
κ_input = I(X ; Z̃(σ)) / d_z
κ_task  = I(Y ; Z̃(σ)) / d_z
```

where

```text
Z̃ = Z + σ · RMS(Z) · ε,   ε ~ N(0, I),   σ = 0.1
```

Mutual information is estimated using an InfoNCE lower bound with fixed batch size `B = 512`.

The information-bottleneck ratio used in the paper is

```text
η_t = κ_task / κ_input
```

The `d_z` normalization cancels algebraically in this ratio, although shared InfoNCE estimator effects may still affect both terms.

---

## Sanity Filter

Records are filtered before analysis using the paper's default Phase 0 filter:

```text
Φ_default = F_main ∧ F_sanity ∧ F_ki_pos

F_main    = exp_type == "main"
F_sanity  = sanity_passed      (κ_permuted < 0.1)
F_ki_pos  = kappa_input > 0
```

All `papers/p*.py` scripts apply this filter consistently. The resulting working set is:

```text
n = 633 records
```

---

## Data

| File | Records | Description |
|---|---:|---|
| `results/full.jsonl` | 1,405 | Main grid plus σ/T/dynamics sweeps |
| `results_p6/full.jsonl` | 836 | Phase-transition width-depth grid |

Total released measurement records:

```text
2,241 records
```

Both JSONL files are released as raw measurement records. The analysis scripts read these files directly.

---

## Project Structure

```text
icon-empirical/
├── manuscripts/
│   └── Information_Capacity_in_Neural_Networks.pdf
│
├── papers/
│   ├── p1_width_law.py
│   ├── p2_depth_law.py
│   ├── p3_physics.py
│   ├── p4_unified.py
│   ├── p5_dynamics.py
│   ├── p6_thermo.py
│   ├── p7_theory.py
│   ├── p8_hw.py
│   └── run_all.py
│
├── core/
│   ├── config.py
│   ├── kappa.py
│   ├── mi_estimator.py
│   ├── noise_channel.py
│   └── dataset.py
│
├── models/
├── experiments/
├── tests/
├── scripts/
├── results/
├── results_p6/
└── requirements.txt
```

---

## Architectures

| Key | Design | Notes |
|---|---|---|
| `mlp` | Fully connected residual network | Baseline feedforward architecture |
| `cnn` | 3-stage ResNet-style CNN | Uses projection to match latent width |
| `gru` | Patch-sequence recurrent model | Sequence-based architecture |
| `transformer_preln` | Pre-LN Transformer | Depth-stable transformer variant |
| `transformer_postln` | Post-LN Transformer | More depth-sensitive variant |
| `serial` | Feedforward stack without residual connections | Shows critical-depth collapse |
| `parallel` | Parallel branches with residual merge | Depth-invariant in Phase 0 grid |
| `boltzmann` | Mean-field RBM-style model | Included as nonstandard architecture family |

---

## Experiment Grid

Main Phase 0 grid:

| Variable | Values |
|---|---|
| Architecture | 8 architecture families |
| Width | 16, 32, 64, 128, 256, 512 |
| Depth | 1, 2, 4, 8, 16, 32 |
| Activation | ReLU, GeLU, tanh |
| Seed | 0, 1, 2 |
| Dataset | CIFAR-10 |

The phase-transition grid focuses on collapse-sensitive architectures with finer depth resolution.

---

## Running from Scratch

Regenerating the raw JSONL files is optional and expensive.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run formula and model tests
python -m pytest tests/

# 3. Smoke test
python tests/pilot.py

# 4. Full experiment run
bash scripts/run.sh all

# 5. Merge shards
python experiments/aggregate.py

# 6. Reproduce paper analyses
python papers/run_all.py --data results/full.jsonl --data_p6 results_p6/full.jsonl
```

For most users, step 6 is sufficient because the raw JSONL files are already included.

---

## Output Format

Each completed condition is stored as one JSON line:

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
  "layerwise": [],
  "epoch_kappas": []
}
```

---

## Reproducibility Notes

The codebase uses fixed seeds for random, NumPy, PyTorch, CUDA, critic initialization, data shuffling, and the measurement noise channel.

The Phase 0 release should be read as a reproducible empirical baseline, not as a universal law claim.

Known scope limits:

- CIFAR-10 only
- Fixed batch size `B = 512`
- InfoNCE finite-batch ceiling effects
- No full held-out critic-evaluation protocol yet
- Architecture and dataset generalization left to Phase 1

---

## How to Contribute

Contributions are welcome, especially:

- Independent reproductions
- Failed reproductions
- Estimator critiques
- Alternative baselines
- New architecture adapters
- New dataset loaders
- Documentation improvements

Please open an issue before starting large changes.

---

## Citation

```bibtex
@misc{park2026information_capacity,
  author = {Park, JoonHa},
  title  = {Information Capacity in Neural Networks: An Empirical Study of Kappa Scaling},
  year   = {2026},
  note   = {Preprint}
}
```

---

## License

This repository is released under the license included in this repository.

---

## Related Repositories

- [`joonhai-official/icon`](https://github.com/joonhai-official/icon) — general ICON framework repository
