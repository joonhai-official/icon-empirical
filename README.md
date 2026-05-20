# icon-empirical

**Reproducibility package for** *Information Capacity in Neural Networks: An Empirical Study of κ Scaling* **(Park, 2026).**

This repository contains the manuscript PDF, raw measurement records, analysis scripts, architecture implementations, and verification utilities for **ICON v0.1.0** — an early open research package for reproducible information-flow measurement in neural representations.

- **Paper PDF:** [`paper/Information_Capacity_in_Neural_Networks.pdf`](paper/Information_Capacity_in_Neural_Networks.pdf)
- **Repository:** <https://github.com/joonhai-official/icon-empirical>
- **Release tag:** [`v0.1.0`](https://github.com/joonhai-official/icon-empirical/releases/tag/v0.1.0)
- **Archived release (Zenodo DOI):** <https://doi.org/10.5281/zenodo.20184000>
- **Companion ICON framework specification:** <https://github.com/joonhai-official/icon> (Zenodo: <https://doi.org/10.5281/zenodo.20184015>)
- **Contact:** joonhai98.official [at] gmail [dot] com

---

## What This Repository Contains

This is a Phase 0 empirical study of κ = MI/d_z scaling in neural representations.

- **Manuscript PDF** of the paper.
- **Raw JSONL measurement records** (2,241 records total) with sha256-verified provenance.
- **Analysis scripts** that map directly to each section of the paper.
- **Architecture implementations and training code** for the eight architecture families studied.

The companion ICON framework specification, which generalizes the protocol used here to a broader measurement framework, lives in the separate [`icon`](https://github.com/joonhai-official/icon) repository.

> **Note on running the code.** This release is structured so that different levels of replication require different levels of compute:
>
> - **Hash-level verification** (no code execution): anyone can verify the released JSONL files byte-for-byte against the SHA256 hashes published below and in §2.7 of the paper.
> - **Analysis-only replication** (CPU or any modest GPU): the analysis scripts under `papers/` read the released JSONL records and recompute the paper's tables and figures. This path does not require retraining and is intended to be runnable on commodity hardware once dependencies are matched.
> - **Partial replication** (single mid-range GPU): individual architecture × width × depth slices can be re-run from scratch on a single GPU. Each individual configuration is small.
> - **Full sweep replication** (multi-GPU or extended single-GPU): regenerating all 2,241 records from scratch corresponds to ≈ 28 GPU-hours of wall time in the original study; this is the most expensive option and is not necessary for verifying the paper's numerical claims.
>
> The author has not personally re-run every path in every environment, so environment-specific issues are possible. Reports of successful or failed runs at any of the levels above are explicitly welcome.

---

## Scope Note — What ICON v0.1.0 Is and Is Not

**ICON v0.1.0 is** an early open measurement proposal anchored by a Phase 0 empirical baseline. It releases code, data, hashes, and manuscripts to support reproduction, criticism, falsification, and extension.

**ICON v0.1.0 is *not*:**

- a finalized standard,
- a claim of universal physical laws,
- a claim of true absolute mutual information capacity,
- a hardware/chip benchmark,
- validated on large foundation models, LLMs, ViTs, or production-scale systems.

All reported phenomena are **within-scope observations under a fixed InfoNCE/noise/protocol regime**. Width scaling (κ ∝ 1/w with R² > 0.9999) is reported as a **phenomenological pattern**, not as a universal law. The InfoNCE finite-batch ceiling at B = 512 (≈ 6.238 nats) and per-dimension normalization are explicit throughout.

---

## Key Phase 0 Observations

| # | Observation | Strength |
|---|---|---|
| 1 | **Critical-depth collapse** in residual-free Serial networks: width-dependent (w=256 → d_c=14; w=512 → d_c=8). Four independent metrics (κ, accuracy, train loss, sanity) agree on d_c. | Most robust |
| 2 | **Task-alignment ratio** η_t = κ_task / κ_input, with exact d_z cancellation, correlates with accuracy at pooled r = 0.685 (n = 633); r > 0.65 in 7 of 8 architectures. | Robust, d_z-canceling |
| 3 | **Training dynamics**: η_t rises within networks during learning (within-config r(η_t, accuracy) ≈ +0.95 across 15 configurations). | Robust within scope |
| 4 | **Forward model κ(w, d)** predicts held-out (w, d) cells within the measured grid with **median 1.75% error** in 5-fold cross-validation; per-architecture models reach ≤ 0.5% in 6 of 8 architectures. | Proof-of-concept (measured grid) |
| 5 | **Phenomenological width pattern**: κ ∝ 1/w with R² > 0.9999 across 8 architectures. | Phenomenological — *not* a universal scaling law |

Each observation is paired with explicit caveats in the paper. The width pattern is reported last because it is the most sensitive to estimator regime (B = 512 InfoNCE ceiling, per-dimension normalization d_z = w).

---

## Hardware-Relevant Interpretation (Read This Before Posting About Chips)

> **ICON v0.1.0 does not benchmark hardware. It also does not pretend hardware is irrelevant.**

The hardware relevance of ICON v0.1.0 comes from the **measured-grid inverse-design result in Section 10** (and its Hardware-Relevant Interpretation in Section 10.6), *not* from direct chip measurements.

What this means in practice:

- ✅ The fitted κ(w, d) model maps a target information-flow density to architectural choices within the measured grid — an empirical proxy for capacity allocation under fixed information-flow targets.
- ✅ This may, if validated further across larger models and tasks, become an empirical basis for capacity-aware architecture sizing and eventually model–accelerator co-design.
- ❌ This release contains **no** chip-level utilization measurements.
- ❌ This release contains **no** accelerator benchmarks (GPU/TPU/NPU/ASIC/FPGA).
- ❌ This release contains **no** energy measurements or memory-bandwidth measurements.
- ❌ This release contains **no** production-scale validation.

If you cite this work in the context of hardware or accelerator design, please cite it as a **measured-grid inverse-design proof of concept**, not as a hardware result. The paper's §13 ("Do not use") states this restriction explicitly.

---

## Data Provenance

The release ships two JSONL files containing the full measurement records.

```
results/full.jsonl       (1,405 records — main sweeps)
results_p6/full.jsonl    (  836 records — phase / dynamics sweeps)
TOTAL                    (2,241 records)
```

### SHA256 hashes

```
results/full.jsonl
  4f1f1959d4c6ad2df3c8e07d7d601636c1a69426295c07db804935fb2b3058a5

results_p6/full.jsonl
  3113f068186e44e0dcc7a4006eb9c415faba43ce648ab86bf303644095d65d42
```

Verify locally with any sha256 tool — no code execution from this repository is required for hash verification. These hashes also appear in §2.7 of the paper.

### Record schema (28 fields)

Each JSONL line is a single measurement record with 28 fields, including κ values, the permuted-pair κ baseline, raw MI in nats, d_z used, training accuracy and loss, sanity flag, saturation flag, wall-clock time, and complete experimental metadata. The full schema is documented in Appendix A of the paper.

---

## Paper-to-Code Mapping

This codebase maps directly onto the paper's structure. Use this table to find the code behind any claim:

| Paper Section | Topic | Analysis Script |
|---|---|---|
| §3 | Activation function effects (canonical config w=256, d=4) | `papers/p1_width_law.py` (activation slice) |
| §4 | Phenomenological width pattern (κ ∝ 1/w) | `papers/p1_width_law.py` |
| §5 | Architecture-specific constants Ĉ_arch | `papers/p1_width_law.py` (C_arch table) |
| §6 | Two-variable scaling κ(w, d) | `papers/p4_unified.py` |
| §7 | Phase-transition-like collapse, critical depth d_c | `papers/p2_depth_law.py`, `papers/p6_thermo.py` |
| §8 | Training dynamics | `papers/p5_dynamics.py` |
| §9 | η_t task-alignment ratio | `papers/p6_thermo.py` |
| §10 | Inverse design (measured-grid proof-of-concept) | `papers/p8_hw.py` |
| §10.6 | Hardware-relevant interpretation | `papers/p8_hw.py` (interpretation only) |
| §11 | Layer-wise patterns | `papers/p2_depth_law.py` (layerwise section) |
| §12.5 | Future direction: hardware-aware information-flow measurement | (discussion only) |
| Appendix B | σ-sweep (robustness) | `papers/p3_physics.py` (σ section) |
| Appendix C | Temperature sweep | `papers/p3_physics.py` (T section) |
| Appendix F | Theory connections (IB, scaling laws) | `papers/p7_theory.py` |

Each script is annotated with a header describing the paper claim it regenerates from the released JSONL data. Independent runs and reports of mismatches are welcome.

---

## Repository Layout

```
icon-empirical/
├── paper/                              # Manuscript PDF
│   └── Information_Capacity_in_Neural_Networks.pdf
├── papers/                             # Analysis scripts (p1 … p8)
├── core/                               # Measurement primitives (κ, MI estimator, noise channel)
├── models/                             # 8 architecture families
├── experiments/                        # Training pipelines
├── results/         results_p6/        # Raw JSONL records + sha256-verified
├── scripts/                            # Setup helpers
├── tests/                              # Test scaffolding (see Note below)
├── requirements.txt
└── README.md
```

> **Note on tests and verification scripts.** Test scaffolding is included in `tests/`. The tests have not been exhaustively run across every Python/torch combination, and some paths (especially those that load full models) will be more sensitive to environment than others. If you encounter environment-specific failures, please open an issue — they will be treated as bugs and patched, not as user error.

---

## What ICON Is — One More Time

ICON v0.1.0 is **an early open research package** for reproducible information-flow measurement in neural representations. It is released openly precisely because the most useful thing the author can do at this stage is invite criticism and replication.

We treat replication and extension as contributions to the open ICON measurement program. Independent reproductions, failed reproductions, criticisms, and extensions are all explicitly welcome.

If you find a result that contradicts ours, please open an issue or send an email — falsifying a Phase 0 baseline is a contribution to ICON, not a competitive act.

---

## Citation

```bibtex
@misc{park2026information_capacity,
  author = {Park, JoonHa},
  title  = {Information Capacity in Neural Networks:
            An Empirical Study of Kappa Scaling},
  year   = {2026},
  note   = {Preprint and reproducibility package.
            Zenodo: https://doi.org/10.5281/zenodo.20184000}
}

@misc{park2026icon,
  author = {Park, JoonHa},
  title  = {ICON: A Framework for Measuring Information Flow},
  year   = {2026},
  note   = {Companion framework specification and reference implementation.
            Zenodo: https://doi.org/10.5281/zenodo.20184015}
}
```

---

## Feedback and Replication

Feedback, criticism, replication attempts, and collaboration inquiries are very welcome. Because the author currently does not have compute available to re-verify the full pipeline in a clean external environment, third-party replication reports — both successful and failed — are particularly valuable at this stage.

- **Contact:** joonhai98.official [at] gmail [dot] com
- **Issues:** <https://github.com/joonhai-official/icon-empirical/issues>
- **Coordinated Phase 1 experiments:** contact the author to discuss collaboration, attribution, and integration of results.

---

## Acknowledgment

This release was prepared as an independent research effort, with Anthropic's Claude used for translation from Korean drafts, editorial refinement, bibliographic verification, LaTeX format conversion, and internal consistency checking. All conceptual content (experimental design, observations, analysis framework, framing decisions) originates with the author, who reviewed all AI-assisted output, verified all numerical claims against raw data, and takes full responsibility for the content.

---

## License

Code is released under the **Apache License 2.0**. Data (JSONL records) is released under **CC BY 4.0**. See `LICENSE` for details.
