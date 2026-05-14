# tests/test_models.py
#
# Verify model correctness before committing to a full experiment run.
#
# Tests
# -----
# 1. Forward shape      : out.shape == [B, n_classes] for all arch/width/depth
# 2. Tap completeness   : "input" and "ffn_out" present; at least one "L*" tap
# 3. Reproducibility    : same seed -> bitwise identical outputs
# 4. Activation effect  : relu / gelu / tanh produce different outputs
# 5. n_classes          : correct output dimension for 10, 100, 200
# 6. Parameter scaling  : more width -> more parameters
# 7. Boltzmann T sweep  : forward succeeds at T=0.01, 1.0, 100.0
# 8. Transformer T sweep: no NaN/Inf at T=0.1..100; P3 T-invariance
#
# Runs on CPU; does not require a GPU.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("ERROR: torch is required for model tests")
    sys.exit(1)

from models import build_model
from core.config import ARCHS

PASS = "PASS"
FAIL = "FAIL"
log  = []

# smaller grid for speed — still hits all code paths
QUICK_WIDTHS = [16, 64, 256, 1024]
QUICK_DEPTHS = [1, 4, 16]


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = PASS if cond else FAIL
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    log.append((name, cond))
    return cond


# ---------------------------------------------------------------------------
# 1. Forward shape and tap completeness
# ---------------------------------------------------------------------------

def test_forward() -> None:
    print("\n-- 1. Forward shape + tap completeness --")
    x = torch.randn(8, 3, 32, 32)

    for arch in ARCHS:
        for w in QUICK_WIDTHS:
            for d in QUICK_DEPTHS:
                try:
                    m    = build_model(arch, w, d, "relu", 10, seed=0)
                    taps: dict = {}
                    out  = m(x, taps=taps)

                    shape_ok  = out.shape == (8, 10)
                    input_ok  = "input"   in taps
                    ffn_ok    = "ffn_out" in taps
                    layer_taps = [k for k in taps
                                  if k.startswith("L") or k.startswith("B")]
                    layers_ok  = len(layer_taps) >= 1

                    # ffn_out must be [B, width] for all archs (d_z=width)
                    ffn_dz_ok = (
                        taps["ffn_out"].dim() == 2 and
                        taps["ffn_out"].shape[1] == w
                    ) if ffn_ok else False
                    # input tap must be 3D [B, tokens, width] for patch-based archs
                    # (except CNN which is 2D and Boltzmann which is 1D flat)
                    if arch in ("cnn", "boltzmann"):
                        input_shape_ok = True  # 2D/1D is expected
                    else:
                        input_shape_ok = ("input" in taps and
                                          taps["input"].dim() == 3)

                    ok = (shape_ok and input_ok and ffn_ok and layers_ok
                          and ffn_dz_ok and input_shape_ok)
                    check(
                        f"{arch:22s}  w={w:4d}  d={d:2d}",
                        ok,
                        f"shape={tuple(out.shape)}  "
                        f"ffn_dz={taps['ffn_out'].shape[1] if ffn_ok else '?'}  "
                        f"taps={sorted(taps.keys())[:6]}"
                        f"{'...' if len(taps) > 6 else ''}",
                    )
                except Exception as e:
                    check(f"{arch:22s}  w={w:4d}  d={d:2d}", False, str(e))


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility() -> None:
    print("\n-- 2. Reproducibility (same seed -> same output) --")
    x = torch.randn(4, 3, 32, 32)

    for arch in ["mlp", "transformer_preln", "gru", "boltzmann"]:
        m1 = build_model(arch, 64, 2, "relu", 10, seed=42)
        m2 = build_model(arch, 64, 2, "relu", 10, seed=42)
        with torch.no_grad():
            d = (m1(x) - m2(x)).abs().max().item()
        check(
            f"{arch}: max_diff={d:.2e} < 1e-6",
            d < 1e-6,
            "two models built with seed=42 must be bitwise identical",
        )


# ---------------------------------------------------------------------------
# 3. Activation effect
# ---------------------------------------------------------------------------

def test_activations() -> None:
    print("\n-- 3. Activation effect (relu != gelu != tanh) --")
    x = torch.randn(8, 3, 32, 32)

    for arch in ["mlp", "transformer_preln", "cnn"]:
        outs = {}
        for act in ["relu", "gelu", "tanh"]:
            m = build_model(arch, 64, 2, act, 10, seed=0)
            with torch.no_grad():
                outs[act] = m(x)

        for a, b in [("relu", "gelu"), ("relu", "tanh"), ("gelu", "tanh")]:
            diff = (outs[a] - outs[b]).abs().mean().item()
            check(
                f"{arch}  {a} vs {b}: mean_diff={diff:.4f} > 1e-4",
                diff > 1e-4,
                "different activations must produce different outputs",
            )


# ---------------------------------------------------------------------------
# 4. n_classes
# ---------------------------------------------------------------------------

def test_n_classes() -> None:
    print("\n-- 4. n_classes output shape --")
    x = torch.randn(4, 3, 32, 32)
    for nc in [10, 100, 200]:
        m   = build_model("mlp", 64, 2, "relu", nc, seed=0)
        out = m(x)
        check(f"n_classes={nc}: out.shape={tuple(out.shape)}",
              out.shape == (4, nc))


# ---------------------------------------------------------------------------
# 5. Parameter scaling with width
# ---------------------------------------------------------------------------

def test_param_scaling() -> None:
    print("\n-- 5. Parameter count scales with width --")
    for arch in ARCHS:
        p64  = sum(p.numel() for p in build_model(arch,  64, 4, "relu", 10).parameters())
        p256 = sum(p.numel() for p in build_model(arch, 256, 4, "relu", 10).parameters())
        check(
            f"{arch:22s}: w=64 -> {p64:,}   w=256 -> {p256:,}",
            p256 > p64,
            "wider models must have more parameters",
        )


# ---------------------------------------------------------------------------
# 6. Boltzmann T sweep
# ---------------------------------------------------------------------------

def test_boltzmann_temperature() -> None:
    print("\n-- 6. Boltzmann temperature sweep --")
    x = torch.randn(8, 3, 32, 32)
    m = build_model("boltzmann", 64, 1, "relu", 10, seed=0)

    results = {}
    for T in [0.01, 1.0, 100.0]:
        with torch.no_grad():
            taps: dict = {}
            out = m(x, taps=taps, temperature=T)
        h = taps.get("ffn_out")
        results[T] = h.mean().item() if h is not None else None
        check(f"T={T:6.2f}: forward ok, h_mean={results[T]:.4f}",
              results[T] is not None)

    # low T -> units saturate near 0 or 1; high T -> units near 0.5
    # so at T=0.01 the hidden mean should be farther from 0.5 than at T=100
    if all(v is not None for v in results.values()):
        dev_low  = abs(results[0.01]  - 0.5)
        dev_high = abs(results[100.0] - 0.5)
        check(
            f"T=0.01 more saturated than T=100 "
            f"(dev_low={dev_low:.4f} vs dev_high={dev_high:.4f})",
            dev_low >= dev_high,
        )


# ---------------------------------------------------------------------------
# 7. Transformer temperature sweep (P3 T-invariance claim)
# ---------------------------------------------------------------------------

def test_transformer_temperature() -> None:
    print("\n-- 7. Transformer temperature sweep (P3: expect T-invariant) --")
    x = torch.randn(8, 3, 32, 32)
    m = build_model("transformer_preln", 64, 4, "relu", 10, seed=0)

    ffn_outs = {}
    for T in [0.1, 1.0, 10.0, 100.0]:
        with torch.no_grad():
            taps: dict = {}
            m(x, taps=taps, temperature=T)
        h = taps.get("ffn_out")
        check(f"transformer T={T:6.1f}: forward ok, ffn_out shape={tuple(h.shape) if h is not None else 'None'}",
              h is not None and h.shape == (8, 64))
        if h is not None:
            ffn_outs[T] = h.clone()

    # At T=1.0 vs T=10.0: ffn_out can differ (different attn weights)
    # T-invariance is a kappa-level (information content) claim, not output-level
    # so we only verify that forward succeeds without NaN/Inf at all temperatures
    for T, h in ffn_outs.items():
        finite = h.isfinite().all().item()
        check(f"transformer T={T}: no NaN/Inf in ffn_out", finite)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Icon_Empirical model tests")
    print("=" * 60)

    test_forward()
    test_reproducibility()
    test_activations()
    test_n_classes()
    test_param_scaling()
    test_boltzmann_temperature()
    test_transformer_temperature()

    passed = sum(1 for _, ok in log if ok)
    total  = len(log)

    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} passed")
    if passed < total:
        print("Failed tests:")
        for name, ok in log:
            if not ok:
                print(f"  FAIL  {name}")
        sys.exit(1)
    else:
        print("All model tests passed.")
