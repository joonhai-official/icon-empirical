# papers/run_all.py
#
# Run all eight analysis scripts in sequence and report a final summary.
#
# This is the last step after experiments/aggregate.py has produced
# results/full.jsonl (and results_p6/full.jsonl for P6).
# Each analysis writes its own JSON output; this script also prints a
# one-line verdict per paper.
#
# Usage
#   python papers/run_all.py --data results/full.jsonl
#                            --data_p6 results_p6/full.jsonl
#                            [--out_dir results/]

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PAPERS = [
    ("p1_width_law", "analysis_p1.json", "Script 1 — Width Law + C_arch (§3, §4, §5)"),
    ("p2_depth_law", "analysis_p2.json", "Script 2 — Depth Law (§7, §11)"),
    ("p3_physics",   "analysis_p3.json", "Script 3 — Robustness sweeps (App. B, App. C)"),
    ("p4_unified",   "analysis_p4.json", "Script 4 — Unified Scaling Law (§6)"),
    ("p5_dynamics",  "analysis_p5.json", "Script 5 — Training Dynamics (§8)"),
    ("p6_thermo",    "analysis_p6.json", "Script 6 — Phase Transitions & IB Ratio (§7, §9)"),
    ("p7_theory",    "analysis_p7.json", "Script 7 — Theory Connections (§2, App. F)"),
    ("p8_hw",        "analysis_p8.json", "Script 8 — Inverse Design (§10)"),
]


# ---------------------------------------------------------------------------
# Verdict functions
# ---------------------------------------------------------------------------

def verdict_p1(path):
    try:
        d  = json.load(open(path))
        r2 = d.get("global", {}).get("r2", 0)
        return f"R2={r2:.4f}  {'SUPPORTED' if r2 >= 0.99 else 'WEAK'}"
    except Exception:
        return "no data"


def verdict_p2(path):
    try:
        d = json.load(open(path))
        s = d.get("depth_sensitive", [])
        i = d.get("depth_invariant", [])
        return f"sensitive={s}  invariant={i}"
    except Exception:
        return "no data"


def verdict_p3(path):
    try:
        d       = json.load(open(path))
        T       = d.get("temperature", {})
        summary = T.get("_arch_summary", {})
        if summary:
            parts = [f"{a}={v['overall_verdict']}" for a, v in sorted(summary.items())]
            return "  |  ".join(parts)
        inv = sum(1 for k, v in T.items()
                  if not k.startswith("_") and v.get("t_invariant"))
        tot = sum(1 for k in T if not k.startswith("_"))
        return f"T_invariant={inv}/{tot}"
    except Exception:
        return "no data"


def verdict_p4(path):
    try:
        d  = json.load(open(path))
        r2 = d.get("kappa_input", {}).get("r2", 0)
        return f"R2={r2:.4f}  {'SUPPORTED' if r2 >= 0.95 else 'WEAK'}"
    except Exception:
        return "no data"


def verdict_p5(path):
    try:
        d   = json.load(open(path))
        kva = d.get("kappa_vs_acc", {})
        r   = kva.get("mean_r_kappa_task", None)
        ok  = kva.get("claim_supported", False)
        return f"r(kappa_task,acc)={r:.3f}  {'SUPPORTED' if ok else 'WEAK'}" if r else "no data"
    except Exception:
        return "no data"


def verdict_p6(path):
    try:
        d = json.load(open(path))
        ki_kt = d.get("ki_kt_dissociation", {})
        verdict = ki_kt.get("verdict", "no verdict")
        return f"ki/kt dissociation: {verdict}"
    except Exception:
        return "no data"


def verdict_p7(path):
    try:
        d    = json.load(open(path))
        conn = d.get("connections_summary", {})
        n    = sum(1 for v in conn.values() if v.get("supported"))
        tot  = len(conn)
        return f"{n}/{tot} connections supported"
    except Exception:
        return "no data"


def verdict_p8(path):
    try:
        d    = json.load(open(path))
        inv  = d.get("inverse_design", {})
        errs = [v.get("kappa_error_pct", 0) for v in inv.values()
                if isinstance(v, dict) and "kappa_error_pct" in v]
        if errs:
            mean_err = sum(errs) / len(errs)
            return f"mean inverse design error: {mean_err:.1f}%"
        return "no data"
    except Exception:
        return "no data"


VERDICTS = [verdict_p1, verdict_p2, verdict_p3, verdict_p4,
            verdict_p5, verdict_p6, verdict_p7, verdict_p8]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="results/full.jsonl")
    p.add_argument("--data_p6", default="results_p6/full.jsonl")
    p.add_argument("--out_dir", default="results/")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    verdicts = []

    for (module, out_file, title), verdict_fn in zip(PAPERS, VERDICTS):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

        # P6 output goes to results_p6/ (alongside its data); all others to out_dir
        if module == "p6_thermo":
            p6_dir   = os.path.dirname(args.data_p6)
            out_path = os.path.join(p6_dir if p6_dir else "results_p6", out_file)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            out_path = os.path.join(args.out_dir, out_file)
        saved_argv = sys.argv[:]

        # P6 writes to results_p6/ by default; P8 also needs P6 data
        if module in ("p6_thermo", "p8_hw"):
            sys.argv = [f"papers/{module}.py",
                        "--data",    args.data,
                        "--data_p6", args.data_p6,
                        "--out",     out_path]
        else:
            sys.argv = [f"papers/{module}.py",
                        "--data", args.data,
                        "--out",  out_path]

        try:
            mod = importlib.import_module(f"papers.{module}")
            importlib.reload(mod)
            mod.main()
            v = verdict_fn(out_path)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            v = f"ERROR: {e}"
        finally:
            sys.argv = saved_argv

        verdicts.append((title, v))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for title, v in verdicts:
        print(f"  {title}")
        print(f"    {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
