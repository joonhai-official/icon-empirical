# ICON

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20184015.svg)](https://doi.org/10.5281/zenodo.20184015)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha%20v0.1.0-orange.svg)](https://github.com/joonhai-official/icon/releases/tag/v0.1.0)

**An early open framework for reproducible information-flow measurement in neural representations.**

ICON provides a small reference implementation for measuring information flow through systems that produce internal numerical representations.

This repository contains the general-purpose ICON framework. The Phase 0 empirical reproduction package is released separately as [`joonhai-official/icon-empirical`](https://github.com/joonhai-official/icon-empirical).

---

## Release and DOI

- GitHub release: [`v0.1.0`](https://github.com/joonhai-official/icon/releases/tag/v0.1.0)
- Zenodo archive: [`10.5281/zenodo.20184015`](https://doi.org/10.5281/zenodo.20184015)
- Empirical reproduction package: [`joonhai-official/icon-empirical`](https://github.com/joonhai-official/icon-empirical)
- Empirical Zenodo archive: [`10.5281/zenodo.20184000`](https://doi.org/10.5281/zenodo.20184000)

---

## Status

This is an **alpha research release**.

ICON v0.1.0 should be read as an early open proposal, not as a finalized standard. Its value depends on independent replication, criticism, and useful extensions.

The goal is to make representation-level information-flow claims more:

- measurable
- reproducible
- falsifiable
- comparable across systems under fixed protocols

Criticism, replications, failed replications, estimator critiques, adapter contributions, and negative results are welcome.

---

## What ICON Defines

ICON defines:

- `F_in` — input information density
- `F_task` — task information density
- `F_self` — self-consistency under a measurement noise channel
- `F_layer` — inter-layer information transmission
- `ρ` — representational dispersion / effective dimensionality
- `η_t = F_task / F_in` — the canonical task-alignment ratio
- `Trust τ` — validity classification for measurements
- `PROTOCOL` — frozen settings and manifest discipline for reproducible measurement

---

## Manuscripts

This repository accompanies the ICON framework specification and companion essay.

- [Icon Framework Specification](manuscripts/Icon_Framework_Specification.pdf)
- [Companion Essay: Information Flow and Self-Reference Across Substrates](manuscripts/Information_Flow_and_Self_Reference_Across_Substrates.pdf)

The empirical Phase 0 paper is hosted in the empirical repository:

- [`icon-empirical`](https://github.com/joonhai-official/icon-empirical)

---

## Repository Roles

ICON is split into separate repositories so the framework, empirical evidence, and future registry can evolve cleanly.

| Repository | Role |
|---|---|
| [`icon`](https://github.com/joonhai-official/icon) | General-purpose framework and reference implementation |
| [`icon-empirical`](https://github.com/joonhai-official/icon-empirical) | Phase 0 empirical paper reproduction package |
| `icon-zoo` | Future public registry of measurement profiles and receipts |
| `icon-site` | Future documentation and project website |

```text
icon            = framework core
icon-empirical  = Phase 0 evidence package
icon-zoo        = future public registry
icon-site       = future website/docs
```

---

## What ICON Measures

At a representation layer `L`, let:

- `X` be the input
- `Y` be the target
- `Z_L` be the clean representation at layer `L`
- `Z̃_L` be the noisy measurement representation
- `Z̃_{L+1}` be the next noisy representation
- `d_L` be the representation dimension

The reference measurement noise channel is:

```text
Z̃_L = Z_L + σ · RMS(Z_L) · ε,    ε ~ N(0, I)
```

ICON computes the following profile:

```text
F_in(L)    = I(X ; Z̃_L) / d_L
F_task(L)  = I(Y ; Z̃_L) / d_L
F_self(L)  = I(Z_L ; Z̃_L) / d_L
F_layer(L) = I(Z̃_L ; Z̃_{L+1}) / d_{L+1}
ρ(L)       = PR(Z_L) / d_L
```

The canonical ratio is:

```text
η_t = F_task / F_in
```

The `d_L` normalization cancels algebraically in `η_t`, although shared estimator effects may still affect both terms.

---

## Why ICON Exists

Modern AI systems are usually compared through aggregate external behavior:

- loss
- accuracy
- benchmark score
- latency
- FLOPs
- parameter count

These are useful, but they do not directly describe how information is represented, preserved, transmitted, concentrated, or lost inside a model.

ICON adds a complementary measurement layer for internal representations. It does not replace probing, CKA, SVCCA, saliency methods, mechanistic interpretability, or other representation-analysis tools.

ICON is designed to help ask:

- How much input information is preserved at this layer?
- How much task-relevant information is present?
- Does the representation survive a controlled noise channel?
- Where does inter-layer information transmission drop?
- Is the representation distributed or collapsed into a small effective subspace?
- Is the measurement valid, saturated, or invalid under the current estimator regime?
- Can the result be reproduced from a manifest and receipt?

---

## Installation

From GitHub:

```bash
pip install git+https://github.com/joonhai-official/icon.git
```

For development:

```bash
git clone https://github.com/joonhai-official/icon
cd icon
pip install -e ".[dev]"
```

For examples requiring vision datasets:

```bash
pip install -e ".[examples]"
```

---

## Quick Start

```python
import icon

# 1. Wrap your system in the AdapterBase contract.
class MyAdapter(icon.AdapterBase):
    def layer_names(self):
        return ["conv1", "conv2", "fc1"]

    def forward_with_taps(self, x, names):
        # Return {name: tensor of shape [N, d_L]} for each requested layer.
        # The reduction from higher-rank activations to [N, d_L] is your choice.
        ...

    def layer_dim(self, name):
        ...

# 2. Wrap your data in the DataLoaderBase contract.
class MyLoader(icon.DataLoaderBase):
    def train_batch(self, batch_size, seed):
        ...

    def val_batch(self, batch_size, seed):
        ...

    def num_classes(self):
        return 10

# 3. Declare a PROTOCOL.
protocol = icon.PROTOCOL()

# 4. Measure.
profile = icon.measure(
    adapter=MyAdapter(model),
    loader=MyLoader(data),
    layer_name="conv2",
    protocol=protocol,
    master_seed=42,
)

# 5. Inspect.
print(profile.components)        # (F_in, F_task, F_self, F_layer, ρ)
print(profile.eta_t)             # F_task / F_in
print(profile.trust)             # per-component Trust τ
print(profile.trust_aggregate)   # profile-level Trust τ
profile.manifest.save("./run.json")
```

---

## Examples

Two examples are included:

- [`examples/mnist_mlp.py`](examples/mnist_mlp.py) — minimal MLP example
- [`examples/cifar10_resnet.py`](examples/cifar10_resnet.py) — ResNet18-style example with forward hooks

Run:

```bash
python examples/mnist_mlp.py
python examples/cifar10_resnet.py
```

The examples are intended to demonstrate the pipeline. Dataset availability and environment details may affect full numerical results.

---

## Tests

Run all tests:

```bash
pytest
```

Run the quick subset:

```bash
pytest -m "not slow"
```

The test suite covers:

- measurement components
- noise injection
- RMS scaling
- Trust τ classification
- participation ratio
- `η_t` identity
- PROTOCOL hashing
- manifest schema
- deterministic measurement behavior
- InfoNCE ceiling behavior

---

## Specification Mapping

The repository follows the framework specification section by section.

| Specification Section | Implementation |
|---|---|
| §1–2: Five measurements | `icon/measurements/` |
| §4: Measurement noise channel | `icon/core/noise.py` |
| §7: Trust τ | `icon/core/trust.py` |
| §9: Measurement pipeline | `icon/pipeline/measure.py` |
| §10: Aggregations | `icon/pipeline/aggregate.py` |
| §11: Cross-system comparison | `icon/pipeline/compare.py` |
| §12: PROTOCOL | `icon/io/protocol.py` |
| §13: Manifest schema | `icon/io/manifest.py` |
| §14: Versioning | `icon/io/manifest.py` |
| §16: Interface contracts | `icon/contracts/` |
| §17: Public surface | `icon/__init__.py` |
| Appendix A.4: InfoNCE | `icon/core/infonce.py` |
| Appendix A.7: η_t identity | `icon/measurements/eta_t.py` |

For a detailed mapping, see [`docs/spec_mapping.md`](docs/spec_mapping.md).

---

## PROTOCOL and Manifests

A measurement without frozen settings is not reproducible.

ICON uses `PROTOCOL` and manifest serialization to record:

- estimator settings
- noise settings
- sample sizes
- seeds
- aggregation choices
- adapter conventions
- environment metadata
- version information

The goal is to make each measurement auditable and reproducible.

A future `icon-zoo` repository will use these manifests and receipts to build a public registry of ICON-compatible measurements.

---

## Interface Contracts

ICON separates the measurement kernel from domain-specific systems.

Users provide:

1. an `AdapterBase` implementation for the model or system
2. a `DataLoaderBase` implementation for the data

This is the main extension point. The core should stay small; domain-specific adapters should grow around it.

Examples of future adapters:

- PyTorch image models
- Hugging Face language models
- audio models
- robotics policies
- graph neural networks
- neuroscience recordings
- signal-processing pipelines

---

## Relationship to the Phase 0 Empirical Paper

The Phase 0 empirical paper is not reproduced in this repository. It has its own repository:

- [`joonhai-official/icon-empirical`](https://github.com/joonhai-official/icon-empirical)

That empirical package contains:

- raw JSONL records
- analysis scripts
- paper-to-code mapping
- empirical manuscript
- Phase 0 reproduction commands

This `icon` repository contains the general measurement framework that the empirical work motivates and supports.

---

## Roadmap

### v0.1.0

- Reference implementation
- Five measurement components
- Trust τ
- PROTOCOL
- manifest serialization
- example adapters
- test suite
- framework specification

### v0.1.x

- Documentation cleanup
- stronger examples
- receipt schema examples
- initial issue templates
- improved environment lock files

### v0.2

- More adapters
- improved estimator diagnostics
- held-out estimator-evaluation utilities
- richer manifest validation
- initial ICON-Zoo seed format

### v0.3+

- ICON-Zoo registry integration
- web viewer / console prototype
- public reproduction reports
- external adapter packages

---

## How to Contribute

Contributions are welcome, especially:

- independent reproductions
- failed reproductions
- estimator critiques
- adapter implementations
- dataset loaders
- manifest / receipt schema improvements
- documentation improvements
- comparisons with CKA, probing, saliency, or other representation-analysis tools

Please open an issue before starting large changes.

Suggested first contributions:

- add a new model adapter
- add a new dataset loader
- test an example in a fresh environment
- improve documentation for Trust τ
- validate a manifest example
- reproduce one Phase 0 empirical result through `icon-empirical`

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citation

Framework:

```bibtex
@misc{park2026icon,
  author = {Park, JoonHa},
  title  = {Icon: A Framework for Measuring Information Flow},
  year   = {2026},
  note   = {Preprint and software release. Zenodo: https://doi.org/10.5281/zenodo.20184015}
}
```

Companion empirical paper:

```bibtex
@misc{park2026information_capacity,
  author = {Park, JoonHa},
  title  = {Information Capacity in Neural Networks: An Empirical Study of Kappa Scaling},
  year   = {2026},
  note   = {Preprint and reproducibility package. Zenodo: https://doi.org/10.5281/zenodo.20184000}
}
```

Companion essay:

```bibtex
@misc{park2026self_reference,
  author = {Park, JoonHa},
  title  = {Information Flow and Self-Reference Across Substrates: On Self-Reference, Substrate, and the Need for Measurement},
  year   = {2026},
  note   = {Companion essay}
}
```

After arXiv submission, citation entries should be updated with arXiv IDs.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Related Repositories

- [`joonhai-official/icon-empirical`](https://github.com/joonhai-official/icon-empirical) — Phase 0 empirical reproduction package
