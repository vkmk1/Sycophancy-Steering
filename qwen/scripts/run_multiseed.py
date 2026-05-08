"""
Multi-seed experiment runner.

Runs the full pipeline (00b -> 02 -> 03) across multiple base-question
subsamples (different seeds), then aggregates results to report mean effect
and cross-seed variance.

Usage:
  python scripts/run_multiseed.py --seeds 42 7 123 456 789
  python scripts/run_multiseed.py --seeds 42 7 123  --split tune

This script orchestrates subprocess calls to the other scripts. It expects
the model to already be loaded by 02_evaluate_steering.py (which it re-runs
for each seed).

After all seeds complete, it reads the per-seed summary files and prints
an aggregate report with mean +/- std across seeds.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from config import ROOT, EVAL_N


def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  FAILED with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run experiment across multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 456, 789],
                        help="Seeds for base-question subsampling")
    parser.add_argument("--split", choices=["tune", "test", "all"], default="all",
                        help="Which split to run (default: all)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip 02_evaluate_steering.py (assume results exist)")
    parser.add_argument(
        "--locked-coefs-from", default=None,
        help="Path to best_coefs_tune.json (typically the aggregated one). "
             "When set and --split test, passes this through to 03_analysis.py "
             "so the test-split selection is NOT re-run on held-out data. "
             "Required for publication-grade tune/test hygiene.",
    )
    parser.add_argument("--skip-response-gen", action="store_true",
                        help="Forward --skip-response-gen to 02_evaluate_steering.py. "
                             "Required to keep the multi-seed sweep within the "
                             "~30 H100-hour budget. Response-token projections are "
                             "captured separately by 02d_response_capture.py at best "
                             "coefs only.")
    args = parser.parse_args()

    seeds = args.seeds
    print(f"Multi-seed run: seeds={seeds}, split={args.split}")

    seed_results = {}

    for seed in seeds:
        print(f"\n\n{'#'*60}")
        print(f"  SEED {seed}")
        print(f"{'#'*60}")

        seed_dir = f"{ROOT}/results/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)

        # 1. Rebuild eval data with this seed
        run_cmd(
            [sys.executable, "00b_rebuild_eval.py", "--seed", str(seed)],
            f"Rebuilding eval_data.json with seed={seed}",
        )

        # 2. Run evaluation (if not skipping)
        if not args.skip_eval:
            eval_cmd = [sys.executable, "02_evaluate_steering.py"]
            if args.split != "all":
                eval_cmd += ["--split", args.split]
            if args.skip_response_gen:
                eval_cmd += ["--skip-response-gen"]
            run_cmd(eval_cmd, f"Evaluating (seed={seed}, split={args.split})")

        # 3. Run analysis
        suffix = f"_{args.split}" if args.split != "all" else ""
        analysis_cmd = [sys.executable, "03_analysis.py"]
        if args.split != "all":
            analysis_cmd += ["--split", args.split]
        if args.locked_coefs_from and args.split == "test":
            analysis_cmd += ["--locked-coefs-from", args.locked_coefs_from]
        run_cmd(analysis_cmd, f"Analyzing (seed={seed}, split={args.split})")

        # 4. Copy results to seed-specific directory (include best_coefs so
        #    aggregate_multiseed.py can build the consensus picks).
        for fname in [f"summary{suffix}.txt",
                      f"statistical_tests{suffix}.json",
                      f"sycophancy_rates{suffix}.json",
                      f"best_coefs{suffix}.json",
                      f"degradation_flags{suffix}.json"]:
            src = f"{ROOT}/results/{fname}"
            dst = f"{seed_dir}/{fname}"
            if os.path.exists(src):
                with open(src) as f:
                    data = f.read()
                with open(dst, "w") as f:
                    f.write(data)

        # 5. Parse key metrics from this seed's results
        rates_path = f"{ROOT}/results/sycophancy_rates{suffix}.json"
        tests_path = f"{ROOT}/results/statistical_tests{suffix}.json"
        if os.path.exists(rates_path) and os.path.exists(tests_path):
            with open(rates_path) as f:
                rates = json.load(f)
            with open(tests_path) as f:
                tests = json.load(f)
            seed_results[seed] = {"rates": rates, "tests": tests}

    # ============================================================
    # Aggregate across seeds
    # ============================================================
    if len(seed_results) < 2:
        print("\nNot enough seed results to aggregate. Done.")
        return

    print(f"\n\n{'='*60}")
    print(f"  MULTI-SEED AGGREGATE ({len(seed_results)} seeds)")
    print(f"{'='*60}")

    # Collect best logit per condition across seeds
    conditions = set()
    for sr in seed_results.values():
        wilcoxon = sr["tests"].get("primary_wilcoxon_best_vs_baseline", {})
        conditions.update(wilcoxon.keys())

    print(f"\n{'Condition':<22}{'mean_logit':>12}{'std':>8}{'mean_p_adj':>12}{'sig_count':>12}")
    print("-" * 66)

    aggregate = {}
    for cond in sorted(conditions):
        logits = []
        padjs = []
        for seed, sr in seed_results.items():
            w = sr["tests"].get("primary_wilcoxon_best_vs_baseline", {}).get(cond, {})
            if w and "best_logit" in w:
                logits.append(w["best_logit"])
                padjs.append(w.get("p_value_adjusted", 1.0))

        if not logits:
            continue

        mean_logit = float(np.mean(logits))
        std_logit = float(np.std(logits))
        mean_padj = float(np.mean(padjs))
        n_sig = sum(1 for p in padjs if p < 0.05)

        aggregate[cond] = {
            "mean_logit": mean_logit,
            "std_logit": std_logit,
            "logits_per_seed": logits,
            "mean_p_adjusted": mean_padj,
            "n_significant": n_sig,
            "n_seeds": len(logits),
        }

        print(f"{cond:<22}{mean_logit:>+12.3f}{std_logit:>8.3f}"
              f"{mean_padj:>12.4f}{n_sig:>8}/{len(logits)}")

    # Save aggregate
    agg_path = f"{ROOT}/results/multiseed_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "split": args.split,
            "per_condition": aggregate,
        }, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")
    print("Done.")


if __name__ == "__main__":
    main()
