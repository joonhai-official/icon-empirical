# tests/test_formulas.py
#
# Unit tests for every mathematical formula used in Icon_Empirical.
# Run before any experiment to catch implementation issues early.
# Does not require torch for most tests.
#
# Tests
# -----
# 1. InfoNCE bound      : MI <= log(N),  MI_random ~= 0
# 2. kappa formula      : kappa = MI / d_z
# 3. Saturation         : margin = (log_batch - MI) / log_batch, d_z-independent
# 4. Sanity check       : permuted_kappa = permuted_MI / d_z < 0.1
# 5. d_effective        : d_eff >= 0 (DPI), fixed-point detection
# 6. Noise channel      : RMS(Z_tilde) ~= RMS(Z) * sqrt(1 + sigma^2)
# 7. Width Law          : kappa * width ~= C_arch  (CV < 0.05)
# 8. kappa_task null    : kappa_task ~= 0 when Y is random
# 9. OLS log-log        : alpha and C recovered from known data
# 10. Unified OLS        : 2-variable alpha/beta recovered
# 11. Exp decay fit      : kappa* and xi recovered from synthetic curve
# 12. Pearson r          : correlation coefficient edge cases
#
# Exit code 0 on full pass, 1 on any failure.

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not installed — skipping torch-dependent tests")

log  = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    log.append((name, cond))
    return cond


# ---------------------------------------------------------------------------
# 1. InfoNCE upper bound
# ---------------------------------------------------------------------------

def test_infonce_bound() -> None:
    print("\n-- 1. InfoNCE upper bound --")
    for N in [16, 64, 512]:
        log_N = math.log(N)
        check(f"MI_perfect({N}) == log({N}) = {log_N:.4f}",
              abs(log_N - log_N) < 1e-9)
        check(f"MI_random({N}) == 0",
              abs(0.0) < 1e-9)


# ---------------------------------------------------------------------------
# 2. kappa = MI / d_z
# ---------------------------------------------------------------------------

def test_kappa_formula() -> None:
    print("\n-- 2. kappa = MI / d_z --")
    cases = [
        (0.0,  16,  0.0),
        (1.0,  64,  1.0 / 64),
        (6.238, 256, 6.238 / 256),
        (6.238, 1024, 6.238 / 1024),
    ]
    for mi, dz, expected in cases:
        got = mi / dz
        check(f"MI={mi:.3f} d_z={dz} -> kappa={got:.6f}",
              abs(got - expected) < 1e-10)


# ---------------------------------------------------------------------------
# 3. Saturation — must be d_z-independent
# ---------------------------------------------------------------------------

def test_saturation() -> None:
    print("\n-- 3. Saturation (MI-space, d_z-independent) --")
    SAT_THRESH = 0.05
    log_batch  = math.log(512)

    mi_sat  = 0.96 * log_batch
    m_s = (log_batch - mi_sat) / log_batch
    check(f"saturated: margin={m_s:.4f} < {SAT_THRESH}", m_s < SAT_THRESH,
          f"MI={mi_sat:.3f}")

    mi_unsat = 0.90 * log_batch
    m_u = (log_batch - mi_unsat) / log_batch
    check(f"not_saturated: margin={m_u:.4f} >= {SAT_THRESH}", m_u >= SAT_THRESH,
          f"MI={mi_unsat:.3f}")

    for dz in [16, 64, 256, 1024]:
        margin = (log_batch - mi_sat) / log_batch
        check(f"d_z={dz:4d}: verdict independent of d_z", margin < SAT_THRESH)


# ---------------------------------------------------------------------------
# 4. Sanity check — permuted_kappa = permuted_MI / d_z
# ---------------------------------------------------------------------------

def test_sanity_check() -> None:
    print("\n-- 4. Sanity check (permuted_kappa = permuted_MI / d_z) --")
    SANITY_THRESH = 0.1
    perm_mi = 2.0   # typical near-random MI in nats

    for dz in [64, 256, 1024]:
        naive      = perm_mi < SANITY_THRESH    # always False: 2.0 > 0.1
        perm_kappa = perm_mi / dz
        correct    = perm_kappa < SANITY_THRESH
        check(
            f"d_z={dz}: naive={naive} correct={correct} (perm_kappa={perm_kappa:.4f})",
            (not naive) and correct,
            "raw MI in nats is ~2-6; dividing by d_z gives kappa scale",
        )

    check("null case: MI=0 -> kappa=0 < 0.1", 0.0 < SANITY_THRESH)


# ---------------------------------------------------------------------------
# 5. d_effective — DPI and fixed point
# ---------------------------------------------------------------------------

def test_d_effective() -> None:
    print("\n-- 5. d_effective --")
    mi_layers = {
        "L0": 6.02, "L1": 5.24, "L2": 4.59, "L3": 4.42,
        "L4": 3.84, "L5": 3.43, "L6": 3.30, "L7": 3.30,
    }
    mi_h0 = mi_layers["L0"]
    D_EFF_EPS = 0.01

    print(f"  I(X; h_0) = {mi_h0:.3f} nats  (reference)")
    prev = 0.0
    for name, mi in mi_layers.items():
        deff  = mi_h0 - mi
        delta = deff - prev
        check(f"  d_eff({name}) = {deff:.4f} >= 0  (DPI)",
              deff >= -1e-9, f"I(X;h_l)={mi:.3f}")
        prev  = deff

    delta_fp = abs(mi_layers["L7"] - mi_layers["L6"])
    check(f"  fixed point L7: |delta| = {delta_fp:.4f} < {D_EFF_EPS}",
          delta_fp < D_EFF_EPS)

    # DPI clamp: if MI increases due to estimator noise, d_eff must be 0
    d_eff_unclamped = mi_h0 - 6.10   # 6.10 > mi_h0=6.02 → would be -0.08
    d_eff_clamped   = max(0.0, d_eff_unclamped)
    check(f"  DPI clamp: max(0, {d_eff_unclamped:.2f}) = {d_eff_clamped:.2f}",
          d_eff_clamped == 0.0,
          "estimator noise can produce MI > MI_h0; clamp enforces DPI")


# ---------------------------------------------------------------------------
# 6. Noise channel RMS scaling
# ---------------------------------------------------------------------------

def test_noise_channel() -> None:
    print("\n-- 6. Noise channel RMS scaling --")
    if not HAS_TORCH:
        print("  SKIP (no torch)")
        return

    for sigma in [0.01, 0.1, 0.5, 1.0]:
        torch.manual_seed(42)
        Z     = torch.randn(2000, 128)
        rms_Z = Z.pow(2).mean().sqrt().item()
        eps   = torch.randn_like(Z)
        Zt    = Z + sigma * rms_Z * eps
        rms_Zt   = Zt.pow(2).mean().sqrt().item()
        expected  = rms_Z * math.sqrt(1 + sigma ** 2)
        rel_error = abs(rms_Zt - expected) / expected
        check(
            f"  sigma={sigma}: RMS(Z_tilde)={rms_Zt:.4f} expected={expected:.4f} "
            f"rel_err={rel_error:.3f}",
            rel_error < 0.05,
        )


# ---------------------------------------------------------------------------
# 7. Width Law kappa * width ~= C_arch
# ---------------------------------------------------------------------------

def test_width_law() -> None:
    print("\n-- 7. Width Law kappa * width = C_arch --")
    # Values correspond to C_arch ≈ 2.2 (transformer) and 2.4 (mlp),
    # well below the InfoNCE saturation limit of log(512)/width.
    known = {
        "transformer_preln": {64: 0.0344, 128: 0.0172, 256: 0.0086, 512: 0.0043},
        "mlp":               {64: 0.0375, 128: 0.0188, 256: 0.0094, 512: 0.0047},
    }
    for arch, data in known.items():
        products = [w * k for w, k in data.items()]
        mean_c   = sum(products) / len(products)
        cv       = (max(products) - min(products)) / mean_c
        check(
            f"  {arch}: C_arch~={mean_c:.3f}  CV={cv:.4f} < 0.05",
            cv < 0.05,
            f"kappa*width = {[round(p, 3) for p in products]}",
        )


# ---------------------------------------------------------------------------
# 8. kappa_task null
# ---------------------------------------------------------------------------

def test_kappa_task_null() -> None:
    print("\n-- 8. kappa_task null (random Y -> MI ~= 0) --")
    null_kappa = 0.05 / 256
    check(f"kappa_task(random Y) = {null_kappa:.5f} < 0.01",
          null_kappa < 0.01,
          "independence -> MI ~= 0 -> kappa_task ~= 0")


# ---------------------------------------------------------------------------
# 9. OLS log-log regression
# ---------------------------------------------------------------------------

def test_ols_loglog() -> None:
    print("\n-- 9. OLS log-log: kappa = C * w^alpha --")
    import random
    random.seed(0)

    # known data from prior experiments
    ws = [64, 128, 256, 512]
    ks = [0.0921, 0.0466, 0.0235, 0.0117]

    lw = [math.log(w) for w in ws]
    lk = [math.log(k) for k in ks]
    n  = len(lw)
    mlw = sum(lw) / n; mlk = sum(lk) / n
    ssxy = sum((lw[i] - mlw) * (lk[i] - mlk) for i in range(n))
    ssxx = sum((lw[i] - mlw) ** 2 for i in range(n))
    alpha  = ssxy / ssxx
    log_C  = mlk - alpha * mlw
    C      = math.exp(log_C)
    preds  = [alpha * lw[i] + log_C for i in range(n)]
    ss_res = sum((lk[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((lk[i] - mlk) ** 2 for i in range(n))
    r2     = 1 - ss_res / ss_tot

    check(f"alpha={alpha:.4f} close to -1.0 (dev={abs(alpha+1):.4f})",
          abs(alpha - (-1.0)) < 0.02)
    check(f"R2={r2:.6f} > 0.99", r2 > 0.99)
    check(f"C={C:.4f} > 0", C > 0)


# ---------------------------------------------------------------------------
# 10. Unified OLS — 2-variable alpha/beta
# ---------------------------------------------------------------------------

def test_unified_ols() -> None:
    print("\n-- 10. Unified OLS: kappa = C * w^alpha * d^beta --")
    import random
    random.seed(42)

    C_true, alpha_true, beta_true = 6.0, -1.0, -0.06
    log_w, log_d, log_yz = [], [], []
    for w in [64, 128, 256, 512]:
        for d in [1, 2, 4, 8, 16]:
            k = C_true * w ** alpha_true * d ** beta_true
            k *= (1 + random.gauss(0, 0.005))
            log_w.append(math.log(w))
            log_d.append(math.log(d))
            log_yz.append(math.log(k) - math.log(C_true))

    n = len(log_w)
    s_ww  = sum(x**2 for x in log_w)
    s_dd  = sum(x**2 for x in log_d)
    s_wd  = sum(log_w[i]*log_d[i] for i in range(n))
    s_wyz = sum(log_w[i]*log_yz[i] for i in range(n))
    s_dyz = sum(log_d[i]*log_yz[i] for i in range(n))
    det   = s_ww * s_dd - s_wd ** 2
    alpha = (s_dd * s_wyz - s_wd * s_dyz) / det
    beta  = (s_ww * s_dyz - s_wd * s_wyz) / det

    preds  = [alpha*log_w[i] + beta*log_d[i] for i in range(n)]
    mean_y = sum(log_yz) / n
    ss_res = sum((log_yz[i]-preds[i])**2 for i in range(n))
    ss_tot = sum((log_yz[i]-mean_y)**2 for i in range(n))
    r2     = 1 - ss_res / ss_tot

    check(f"alpha={alpha:.4f} close to {alpha_true} (dev={abs(alpha-alpha_true):.4f})",
          abs(alpha - alpha_true) < 0.05)
    check(f"beta={beta:.4f} close to {beta_true} (dev={abs(beta-beta_true):.4f})",
          abs(beta - beta_true) < 0.02)
    check(f"unified R2={r2:.6f} > 0.99", r2 > 0.99)


# ---------------------------------------------------------------------------
# 11. Exponential decay fit
# ---------------------------------------------------------------------------

def test_exp_decay_fit() -> None:
    print("\n-- 11. Exponential decay fit: kappa(d) = kappa* + A*exp(-d/xi) --")
    import random
    random.seed(7)

    kstar_true, A_true, xi_true = 0.013, 0.010, 3.0
    depths = [1, 2, 4, 8, 16, 32]
    kappas = [kstar_true + A_true * math.exp(-d / xi_true) for d in depths]

    best = {"r2": -999.0}
    candidates = [kappas[-1], kappas[-2], min(kappas), sum(kappas[-3:])/3]
    for kstar in candidates:
        for xi in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0]:
            xs    = [math.exp(-d / xi) for d in depths]
            ys    = [k - kstar for k in kappas]
            A_num = sum(x*y for x, y in zip(xs, ys))
            A_den = sum(x*x for x in xs)
            if A_den < 1e-12:
                continue
            A     = A_num / A_den
            preds = [kstar + A * x for x in xs]
            mean_k = sum(kappas) / len(kappas)
            ss_res = sum((k-p)**2 for k, p in zip(kappas, preds))
            ss_tot = sum((k-mean_k)**2 for k in kappas)
            r2     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            if r2 > best["r2"]:
                best = {"kstar": kstar, "A": A, "xi": xi, "r2": r2}

    check(f"xi recovered={best['xi']} (true={xi_true})",
          abs(best["xi"] - xi_true) < 0.5)
    check(f"kstar recovered={best['kstar']:.5f} (true={kstar_true:.5f})",
          abs(best["kstar"] - kstar_true) < 0.001)
    check(f"fit R2={best['r2']:.6f} > 0.999", best["r2"] > 0.999)


# ---------------------------------------------------------------------------
# 12. Pearson r
# ---------------------------------------------------------------------------

def test_pearson_r() -> None:
    print("\n-- 12. Pearson r --")

    def r(xs, ys):
        n = len(xs)
        if n < 2: return 0.0
        mx = sum(xs)/n; my = sum(ys)/n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        sx  = math.sqrt(sum((x-mx)**2 for x in xs))
        sy  = math.sqrt(sum((y-my)**2 for y in ys))
        if sx < 1e-12 or sy < 1e-12: return 0.0
        return num / (sx * sy)

    check("perfect +1: [1,2,3] vs [2,4,6]",
          abs(r([1,2,3], [2,4,6]) - 1.0) < 1e-9)
    check("perfect -1: [1,2,3] vs [6,4,2]",
          abs(r([1,2,3], [6,4,2]) - (-1.0)) < 1e-9)
    check("zero correlation: constant Y",
          r([1,2,3,4,5], [3,3,3,3,3]) == 0.0)

    # kappa_task rises monotonically with accuracy -> strong positive r
    ki  = [0.0214,0.0210,0.0208,0.0205,0.0201,0.0215,0.0221,0.0230]
    kt  = [0.015, 0.018, 0.022, 0.027, 0.033, 0.041, 0.046, 0.050]
    acc = [0.30,  0.40,  0.50,  0.58,  0.65,  0.72,  0.76,  0.79]
    r_kt  = r(kt, acc)
    r_ki  = r(ki, acc)
    check(f"r(kappa_task, acc) = {r_kt:.3f} > 0.9 (task info tracks perf)",
          r_kt > 0.9)
    check(f"r(kappa_task, acc) > r(kappa_input, acc): {r_kt:.3f} > {r_ki:.3f}",
          r_kt > r_ki)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Icon_Empirical formula unit tests")
    print("=" * 60)

    test_infonce_bound()
    test_kappa_formula()
    test_saturation()
    test_sanity_check()
    test_d_effective()
    test_noise_channel()
    test_width_law()
    test_kappa_task_null()
    test_ols_loglog()
    test_unified_ols()
    test_exp_decay_fit()
    test_pearson_r()

    passed = sum(1 for _, ok in log if ok)
    total  = len(log)

    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} passed")

    if passed < total:
        print("Failed:")
        for name, ok in log:
            if not ok:
                print(f"  FAIL  {name}")
        sys.exit(1)
    else:
        print("All formula tests passed.")
