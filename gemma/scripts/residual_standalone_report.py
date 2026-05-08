"""Post-run reporting for the residual_standalone conditions.

Produces:
  - figures/fig7_residual_standalone_vs_parent.png: coefficient-sweep overlays
    (residual, parent role, CAA) on shared axes, matching the fig1 style.
  - results/residual_standalone_report.csv: one row per condition in the same
    format as the existing results tables, with:
      condition, condition_type, best_coef, parent_role, parent_best_coef,
      coef_differs_from_parent, baseline_logit, best_logit, delta_logit,
      delta_logit_ci_lo, delta_logit_ci_hi, baseline_rate, best_rate,
      delta_rate_pp, delta_rate_ci_lo_pp, delta_rate_ci_hi_pp,
      wilcoxon_p_raw, wilcoxon_p_holm, significant_after_mcc, alternative,
      cosine_with_caa (of parent role).
  - results/residual_standalone_summary.txt: short plain-text report.

This reads results/*_test.json produced by 02_evaluate_steering.py + the
Holm-corrected p-values in results/statistical_tests_test.json (after
03_analysis.py has re-run Holm across the 24-condition family). It does NOT
re-compute Holm itself -- that comes from 03_analysis.py's apply_mcc on the
full family.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ROOT, COEFFICIENTS, COLORS, LABELS,
    CONDITIONS_RESIDUAL_STANDALONE, RESIDUAL_STANDALONE_ROLES,
    CONDITION_TYPE,
)


def bootstrap_ci(rows, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx_ = rng.choice(len(rows), size=len(rows), replace=True)
        vals.append(fn([rows[i] for i in idx_]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff_ci(steered_rows, baseline_rows, metric, n=1000, seed=0):
    """Bootstrap 95% CI on the mean of per-BASE paired differences.

    metric in {"logit", "rate"}. We aggregate to base-question level first
    (averaging the two variant orderings), then take the paired difference
    (steered - baseline) for each base, then bootstrap the mean of those
    pair differences. This matches the resolution of the Wilcoxon test
    rather than pooling rows independently, so the CI accounts for
    within-base correlation."""
    from collections import defaultdict

    def base_agg(rows, metric):
        by_base = defaultdict(list)
        for r in rows:
            v = r["syc_logit"] if metric == "logit" else (1.0 if r["chose_sycophantic"] else 0.0)
            by_base[r["base_id"]].append(v)
        return {b: float(np.mean(vs)) for b, vs in by_base.items()}

    s_map = base_agg(steered_rows, metric)
    b_map = base_agg(baseline_rows, metric)
    common = sorted(set(s_map) & set(b_map))
    if not common:
        return 0.0, 0.0, 0.0
    diffs = np.array([s_map[b] - b_map[b] for b in common])
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for i in range(n):
        idx_ = rng.integers(0, len(diffs), len(diffs))
        boot[i] = diffs[idx_].mean()
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return observed, lo, hi


def rate_of(rows):
    return sum(1 for r in rows if r["chose_sycophantic"]) / len(rows) if rows else 0.0


def logit_of(rows):
    return float(np.mean([r["syc_logit"] for r in rows])) if rows else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["tune", "test"])
    ap.add_argument(
        "--parent-coefs-from", default=None,
        help="best_coefs_tune.json used to lookup parent-role coefficients. "
             "Defaults to {ROOT}/results/best_coefs_tune.json.",
    )
    args = ap.parse_args()
    suffix = f"_{args.split}"

    with open(f"{ROOT}/results/all_results{suffix}.json") as f:
        all_rows = json.load(f)
    with open(f"{ROOT}/results/sycophancy_rates{suffix}.json") as f:
        rates = json.load(f)
    with open(f"{ROOT}/results/best_coefs{suffix}.json") as f:
        best_coefs = json.load(f)["best_coefs"]
    with open(f"{ROOT}/results/statistical_tests{suffix}.json") as f:
        tests = json.load(f)
    # Parent-role coefficients are locked on the tune split.
    parent_path = args.parent_coefs_from or f"{ROOT}/results/best_coefs_tune.json"
    with open(parent_path) as f:
        parent_best_coefs = json.load(f)["best_coefs"]
    with open(f"{ROOT}/vectors/steering/caa_decomposition.json") as f:
        decomp = json.load(f)

    idx = defaultdict(list)
    for r in all_rows:
        idx[(r["condition"], r["coefficient"])].append(r)

    # A canonical baseline row set for delta computations
    any_real = [c for c in rates if not c.startswith("random_")][0]
    baseline_rows = idx[(any_real, 0.0)]
    baseline_rate = rate_of(baseline_rows)
    baseline_logit = logit_of(baseline_rows)

    # ---------- Figure: residual vs parent vs CAA sweeps ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), sharex=True)
    for col, parent in enumerate(RESIDUAL_STANDALONE_ROLES):
        resid = f"{parent}_residual"
        series = [
            ("caa",    "CAA (targeted)",    COLORS["caa"]),
            (parent,   LABELS[parent],      COLORS[parent]),
            (resid,    LABELS[resid],       COLORS[resid]),
        ]
        ax_rate = axes[0, col]
        ax_log  = axes[1, col]
        for cond, lab, color in series:
            ys_rate, ys_log = [], []
            for c in COEFFICIENTS:
                k = f"{c}" if c != 0.0 else "0.0"
                s = rates.get(cond, {}).get(k)
                ys_rate.append(s["binary_rate"] * 100 if s else float("nan"))
                ys_log.append(s["mean_syc_logit"] if s else float("nan"))
            style = dict(color=color, label=lab, marker="o", markersize=4,
                         lw=2.4 if cond == resid else 1.6,
                         linestyle="-" if cond != "caa" else "--")
            ax_rate.plot(COEFFICIENTS, ys_rate, **style)
            ax_log.plot(COEFFICIENTS, ys_log, **style)

        ax_rate.axhline(baseline_rate * 100, ls=":", color="black", alpha=0.7,
                        label="baseline")
        ax_rate.set_title(f"{LABELS[parent]}: residual vs parent vs CAA")
        ax_rate.set_ylabel("Sycophancy rate (%)" if col == 0 else "")
        ax_rate.grid(True, alpha=0.3)
        ax_rate.legend(loc="best", fontsize=8)

        ax_log.axhline(baseline_logit, ls=":", color="black", alpha=0.7,
                       label="baseline")
        ax_log.axhline(0, ls="-", color="black", lw=0.6, alpha=0.5)
        ax_log.set_xlabel("Steering coefficient")
        ax_log.set_ylabel("Mean sycophancy logit" if col == 0 else "")
        ax_log.grid(True, alpha=0.3)
        ax_log.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Fig 7: Standalone CAA-orthogonal residual vs parent role vs CAA "
        f"(coefficient sweep, {args.split} split)",
        fontsize=12,
    )
    plt.tight_layout()
    out_fig = f"{ROOT}/figures/fig7_residual_standalone_vs_parent.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_fig}")

    # ---------- CSV + per-condition summary ----------
    wilcoxon_map = tests.get("primary_wilcoxon_best_vs_baseline", {})
    csv_rows = []
    summary_lines = [
        "=== Standalone CAA-orthogonal residual conditions ===",
        f"Split: {args.split}    Baseline logit: {baseline_logit:+.3f}    "
        f"Baseline rate: {baseline_rate*100:.1f}%",
        f"Parent-role tune coefficients read from: {parent_path}",
        "",
    ]

    for parent in RESIDUAL_STANDALONE_ROLES:
        cond = f"{parent}_residual"
        best_c = float(best_coefs.get(cond, 0.0))
        best_rows = idx.get((cond, best_c), [])
        if not best_rows:
            print(f"WARN: no data for {cond} at coef {best_c}")
            continue
        parent_best_c = float(parent_best_coefs.get(parent, 0.0))

        # Observed effects at locked coefficient (vs baseline coef=0 rows)
        best_logit = logit_of(best_rows)
        best_rate = rate_of(best_rows)
        d_logit, d_logit_lo, d_logit_hi = paired_diff_ci(
            best_rows, baseline_rows, "logit", n=2000, seed=42,
        )
        d_rate_obs, d_rate_lo, d_rate_hi = paired_diff_ci(
            best_rows, baseline_rows, "rate", n=2000, seed=42,
        )

        w = wilcoxon_map.get(cond, {})
        p_raw = w.get("p_value_raw")
        p_adj = w.get("p_value_adjusted")
        sig = w.get("significant_after_mcc")
        alt = w.get("alternative", "two-sided")
        cos_caa_parent = decomp.get(parent, {}).get("cosine_with_caa")

        csv_rows.append({
            "condition": cond,
            "condition_type": CONDITION_TYPE[cond],
            "parent_role": parent,
            "parent_cosine_with_caa": cos_caa_parent,
            "best_coef": best_c,
            "parent_best_coef_tune": parent_best_c,
            "coef_differs_from_parent": best_c != parent_best_c,
            "baseline_logit": baseline_logit,
            "best_logit": best_logit,
            "delta_logit": d_logit,
            "delta_logit_ci_lo": d_logit_lo,
            "delta_logit_ci_hi": d_logit_hi,
            "baseline_rate": baseline_rate,
            "best_rate": best_rate,
            "delta_rate_pp": d_rate_obs * 100,
            "delta_rate_ci_lo_pp": d_rate_lo * 100,
            "delta_rate_ci_hi_pp": d_rate_hi * 100,
            "wilcoxon_alternative": alt,
            "wilcoxon_p_raw": p_raw,
            "wilcoxon_p_holm": p_adj,
            "significant_after_holm": sig,
        })

        def fmt_p(p):
            if p is None:
                return "n/a"
            if p < 1e-15:
                return "< 1e-15"
            return f"{p:.4g}"

        summary_lines += [
            f"--- {LABELS[cond]} (|cos({parent}, CAA)| = {abs(cos_caa_parent):.4f}) ---",
            f"  best coef (tune-locked, {args.split}): {best_c:+.0f}    "
            f"parent ({parent}) best coef: {parent_best_c:+.0f}    "
            f"{'DIFFERS' if best_c != parent_best_c else 'same'}",
            f"  delta logit (vs baseline): {d_logit:+.3f}  "
            f"[{d_logit_lo:+.3f}, {d_logit_hi:+.3f}]",
            f"  delta rate (pp)          : {d_rate_obs*100:+.1f}  "
            f"[{d_rate_lo*100:+.1f}, {d_rate_hi*100:+.1f}]",
            f"  Wilcoxon ({alt}): p_raw = {fmt_p(p_raw)}  "
            f"p_holm (24-condition family) = {fmt_p(p_adj)}  "
            f"{'*sig*' if sig else 'ns'}",
            "",
        ]

    csv_path = f"{ROOT}/results/residual_standalone_report_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"saved {csv_path}")

    # ---------- Pattern check ----------
    # Pre-existing claim (matched-coefficient decomposition): low-cosine
    # residuals reduce syc, the high-cosine residual does not. Apply the
    # user's rule relative to THIS experiment's residuals: the role with the
    # largest |cos(role, CAA)| is the "high" residual, the others are "low".
    by_cos = sorted(csv_rows, key=lambda r: abs(r["parent_cosine_with_caa"]))
    low_cos_rows = by_cos[:-1]
    high_cos_rows = by_cos[-1:]

    def _reduced(row):
        # Reduction = negative delta logit whose 95% CI excludes 0 (paired bootstrap).
        # AND Holm-corrected significance.
        return (
            row["delta_logit"] < 0
            and row["delta_logit_ci_hi"] < 0
            and row.get("significant_after_holm") is True
        )

    summary_lines += [
        "--- Pattern check vs matched-coefficient decomposition finding ---",
        "Claim under test: low-|cos(role,CAA)| residuals reduce sycophancy; "
        "high-|cos(role,CAA)| residual does not.",
        "",
        f"Low-cosine residuals ({', '.join(r['parent_role'] for r in low_cos_rows)}):",
    ]
    for r in low_cos_rows:
        summary_lines.append(
            f"  {r['condition']}: delta_logit={r['delta_logit']:+.3f} "
            f"[{r['delta_logit_ci_lo']:+.3f}, {r['delta_logit_ci_hi']:+.3f}]  "
            f"p_holm={r.get('wilcoxon_p_holm')}  "
            f"reduces={'yes' if _reduced(r) else 'NO'}"
        )
    summary_lines += ["", f"High-cosine residual ({', '.join(r['parent_role'] for r in high_cos_rows)}):"]
    for r in high_cos_rows:
        summary_lines.append(
            f"  {r['condition']}: delta_logit={r['delta_logit']:+.3f} "
            f"[{r['delta_logit_ci_lo']:+.3f}, {r['delta_logit_ci_hi']:+.3f}]  "
            f"p_holm={r.get('wilcoxon_p_holm')}  "
            f"reduces={'yes' if _reduced(r) else 'NO'}"
        )

    low_all_reduce = all(_reduced(r) for r in low_cos_rows)
    high_any_reduce = any(_reduced(r) for r in high_cos_rows)
    verdict = (
        "PATTERN HOLDS" if (low_all_reduce and not high_any_reduce)
        else "PATTERN DOES NOT HOLD"
    )
    summary_lines += ["", f"Verdict: {verdict}",
                      f"  (low_all_reduce={low_all_reduce}  "
                      f"high_any_reduce={high_any_reduce})"]

    summary_path = f"{ROOT}/results/residual_standalone_summary_{args.split}.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"saved {summary_path}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
