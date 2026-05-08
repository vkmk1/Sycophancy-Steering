"""
Aggregate per-seed results from run_multiseed.py into a per-split summary
readable by the paper figures.

Reads: results/seed_<seed>/{sycophancy_rates_<split>.json,
                           statistical_tests_<split>.json,
                           best_coefs_<split>.json (if present)}.

Writes: results/multiseed_aggregate_<split>.json with structure:
    {
      "split": "test",
      "seeds": [42, 7, 123],
      "n_seeds": 3,
      "per_condition": {
        "skeptic": {
            "best_coefs": [2000.0, 2000.0, 1000.0],
            "logits":     [ 0.31,  0.29,   0.33],
            "rates":      [ 0.50,  0.48,   0.51],
            "mean_logit": 0.31,
            "std_logit":  0.018,
            "mean_rate":  0.496,
            "std_rate":   0.0125,
            "n_seeds":    3,
            "wilcoxon_p_adj": [1e-20, 5e-19, 1e-18],
            "n_significant": 3,
        }, ...
      }
    }
"""
import argparse
import json
import os
import glob
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
# Qwen copy: aggregate_multiseed.py lives directly in scripts/, not in paper/scripts/
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from config import ROOT, CONDITIONS_REAL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test", "all"], default="test")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Seeds to aggregate. Defaults to all seed_* dirs found.")
    args = ap.parse_args()
    suffix = f"_{args.split}" if args.split != "all" else ""

    res_dir = f"{ROOT}/results"
    if args.seeds is None:
        seed_dirs = sorted(glob.glob(f"{res_dir}/seed_*"))
        seeds = [int(os.path.basename(d).split("_", 1)[1]) for d in seed_dirs]
    else:
        seeds = list(args.seeds)
    if not seeds:
        print("No seed_* directories found; nothing to aggregate.")
        return
    print(f"Aggregating seeds: {seeds} for split={args.split}")

    per_cond = {c: {"best_coefs": [], "logits": [], "rates": [],
                    "wilcoxon_p_adj": [], "wilcoxon_significant": []}
                for c in CONDITIONS_REAL}

    missing = 0
    for s in seeds:
        rates_path = f"{res_dir}/seed_{s}/sycophancy_rates{suffix}.json"
        tests_path = f"{res_dir}/seed_{s}/statistical_tests{suffix}.json"
        bc_path    = f"{res_dir}/seed_{s}/best_coefs{suffix}.json"
        if not (os.path.exists(rates_path) and os.path.exists(tests_path)):
            print(f"  WARN: seed {s}: missing rates/tests JSONs (split={args.split})")
            missing += 1
            continue
        with open(rates_path) as f:
            rates = json.load(f)
        with open(tests_path) as f:
            tests = json.load(f)
        best_coefs = {}
        if os.path.exists(bc_path):
            with open(bc_path) as f:
                best_coefs = json.load(f).get("best_coefs", {})
        for cond in CONDITIONS_REAL:
            w = tests.get("primary_wilcoxon_best_vs_baseline", {}).get(cond)
            if not w:
                continue
            bc = best_coefs.get(cond, w.get("best_coef"))
            if bc is None:
                continue
            k = f"{bc}" if bc != 0.0 else "0.0"
            s_cell = rates.get(cond, {}).get(k)
            if not s_cell:
                continue
            per_cond[cond]["best_coefs"].append(float(bc))
            per_cond[cond]["logits"].append(float(s_cell["mean_syc_logit"]))
            per_cond[cond]["rates"].append(float(s_cell["binary_rate"]))
            per_cond[cond]["wilcoxon_p_adj"].append(float(w.get("p_value_adjusted", 1.0)))
            per_cond[cond]["wilcoxon_significant"].append(bool(w.get("significant_after_mcc")))

    out = {"split": args.split, "seeds": seeds, "n_seeds": len(seeds),
           "per_condition": {}, "missing_seeds": missing}
    for cond, d in per_cond.items():
        if not d["logits"]:
            continue
        out["per_condition"][cond] = {
            "best_coefs": d["best_coefs"],
            "logits": d["logits"],
            "rates": d["rates"],
            "mean_logit": float(np.mean(d["logits"])),
            "std_logit": float(np.std(d["logits"], ddof=1)) if len(d["logits"]) > 1 else 0.0,
            "mean_rate": float(np.mean(d["rates"])),
            "std_rate": float(np.std(d["rates"], ddof=1)) if len(d["rates"]) > 1 else 0.0,
            "n_seeds": len(d["logits"]),
            "wilcoxon_p_adj": d["wilcoxon_p_adj"],
            "n_significant": int(sum(d["wilcoxon_significant"])),
        }
        print(f"  {cond:<20} n={len(d['logits'])}  "
              f"logit={out['per_condition'][cond]['mean_logit']:+.3f} "
              f"±{out['per_condition'][cond]['std_logit']:.3f}  "
              f"sig {out['per_condition'][cond]['n_significant']}/"
              f"{out['per_condition'][cond]['n_seeds']}")

    out_path = f"{res_dir}/multiseed_aggregate{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # On tune-split aggregate, write a consensus best_coefs file that the
    # multi-seed test runs can lock on. Consensus rule: median coefficient
    # per condition across seeds. If the median lies between sweep points,
    # round toward the coef that appears more often; break ties toward the
    # one with the best mean excess over random.
    if args.split == "tune":
        import statistics as _st
        consensus = {}
        for cond, d in out["per_condition"].items():
            cs = d.get("best_coefs", [])
            if not cs:
                continue
            # Pick the mode first; if no unique mode, pick the median and
            # snap to the nearest coefficient that actually appears in cs.
            try:
                m = _st.mode(cs)
            except _st.StatisticsError:
                med = _st.median(cs)
                m = min(cs, key=lambda c: (abs(c - med), -cs.count(c)))
            consensus[cond] = float(m)
        consensus_path = f"{res_dir}/best_coefs_tune_aggregate.json"
        with open(consensus_path, "w") as f:
            json.dump({
                "source_split": "tune",
                "seeds": seeds,
                "rule": "mode across seeds, tie-break by count then proximity to median",
                "best_coefs": consensus,
            }, f, indent=2)
        print(f"Wrote {consensus_path}  (use as --locked-coefs-from for test runs)")


if __name__ == "__main__":
    main()
