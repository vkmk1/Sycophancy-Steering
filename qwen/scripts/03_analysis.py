"""
Script 3: Analysis -- figures and statistical tests.

Key improvements over original:
  - All primary analyses at BASE-QUESTION level (n=300 or n=split), not row level.
  - Spearman dose-response computed on ROW-LEVEL data, not 6 aggregate means.
  - Pearson projection-vs-sycophancy pooled across steered conditions.
  - Multiple comparison correction (Holm-Bonferroni by default).
  - Exact McNemar when off-diagonal cells < 5.
  - Random control summarised as MEAN across N_RANDOM_VECTORS random vectors.
  - p-value underflow guarded (never reports exactly 0.0).
  - Primary significance test: paired Wilcoxon signed-rank on base-level syc_logit.
  - Confounded row-level logistic regression removed from main results.
  - Supports --split tune|test|all for tune/test workflow.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import torch

from config import (
    ROOT, CONDITIONS, CONDITIONS_REAL, CONDITIONS_RANDOM,
    COEFFICIENTS, COLORS, LABELS, TARGET_LAYER,
    CRITICAL_ROLES, CONFORMIST_ROLES,
    N_RANDOM_VECTORS, MCC_METHOD,
    DEGRAD_RATE_TOL, DEGRAD_LOGIT_TOL,
    EXPECTED_DIRECTION,
)

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["tune", "test", "all"], default="all")
parser.add_argument(
    "--locked-coefs-from",
    default=None,
    help="Path to a best_coefs_*.json produced by a prior --split tune run. "
         "When set, skip best-coef selection on this split and use the locked "
         "coefficients (required for clean tune/test hygiene on --split test).",
)
args = parser.parse_args()
suffix = f"_{args.split}" if args.split != "all" else ""

FIG_DIR = f"{ROOT}/figures"
RES_DIR = f"{ROOT}/results"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Utility ----

def safe_pvalue(p):
    """Guard against p-value underflow: never report exactly 0.0."""
    if p == 0.0:
        return 1e-300  # representable floor; display as "< 1e-15"
    return float(p)


def fmt_p(p):
    """Format p-value for display, handling underflow."""
    if p < 1e-15:
        return "< 1e-15"
    return f"{p:.4g}"


def apply_mcc(pvals, method=MCC_METHOD):
    """Apply multiple comparison correction. Returns (reject, adjusted_pvals)."""
    if len(pvals) == 0:
        return [], []
    reject, adj, _, _ = multipletests(pvals, method=method)
    return reject.tolist(), adj.tolist()


# ---- Load data ----
print(f"Loading data ({args.split} split)...")

with open(f"{RES_DIR}/all_results{suffix}.json") as f:
    all_results = json.load(f)
with open(f"{RES_DIR}/sycophancy_rates{suffix}.json") as f:
    rates = json.load(f)
with open(f"{RES_DIR}/vector_cosine_similarities.json") as f:
    cos = json.load(f)
with open(f"{RES_DIR}/degradation_flags{suffix}.json") as f:
    degraded = json.load(f)

# Index: (condition, coeff) -> list of rows
idx = defaultdict(list)
for r in all_results:
    idx[(r["condition"], r["coefficient"])].append(r)

# Use first real condition's baseline as canonical
baseline_rows = idx[(CONDITIONS_REAL[0], 0.0)]
baseline_rate = sum(1 for r in baseline_rows if r["chose_sycophantic"]) / len(baseline_rows)
baseline_logit = float(np.mean([r["syc_logit"] for r in baseline_rows]))
n_base = len(set(r["base_id"] for r in baseline_rows))
print(f"Baseline N={len(baseline_rows)} rows ({n_base} base questions)  "
      f"binary={baseline_rate*100:.1f}%  mean_syc_logit={baseline_logit:+.4f}")


def rate_of(rows):
    return sum(1 for r in rows if r["chose_sycophantic"]) / len(rows) if rows else 0.0


def logit_of(rows):
    return float(np.mean([r["syc_logit"] for r in rows])) if rows else 0.0


def base_level_logits(rows):
    """Average syc_logit per base question across both variants."""
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r["syc_logit"])
    return {b: np.mean(vs) for b, vs in by_base.items()}


def bootstrap_ci(rows, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = rng.choice(len(rows), size=len(rows), replace=True)
        vals.append(fn([rows[i] for i in sample]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- Random-mean at each coefficient ----
def random_mean_at_coeff(coeff_key):
    """Mean syc_logit across all random vectors at a given coefficient."""
    vals = []
    for rc in CONDITIONS_RANDOM:
        s = rates.get(rc, {}).get(coeff_key)
        if s:
            vals.append(s["mean_syc_logit"])
    return np.mean(vals) if vals else None


# ---- Figure 1: Sycophancy vs steering coefficient ----
print("\n--- Figure 1 ---")
fig1, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 10))

# Show real conditions + random_mean
for cond in CONDITIONS_REAL:
    ys_rate, ys_logit = [], []
    for c in COEFFICIENTS:
        k = f"{c}" if c != 0.0 else "0.0"
        s = rates[cond][k]
        ys_rate.append(s["binary_rate"] * 100)
        ys_logit.append(s["mean_syc_logit"])
    style = dict(color=COLORS[cond], label=LABELS[cond], marker="o", markersize=4)
    if cond == "assistant_axis":
        style.update(lw=2.8)
    ax_a.plot(COEFFICIENTS, ys_rate, **style)
    ax_b.plot(COEFFICIENTS, ys_logit, **style)

# Random: plot mean +/- std across N_RANDOM_VECTORS
rand_rates_by_coeff, rand_logits_by_coeff = [], []
for c in COEFFICIENTS:
    k = f"{c}" if c != 0.0 else "0.0"
    rs = [rates[rc][k]["binary_rate"] * 100 for rc in CONDITIONS_RANDOM if k in rates.get(rc, {})]
    ls = [rates[rc][k]["mean_syc_logit"] for rc in CONDITIONS_RANDOM if k in rates.get(rc, {})]
    rand_rates_by_coeff.append((np.mean(rs), np.std(rs)))
    rand_logits_by_coeff.append((np.mean(ls), np.std(ls)))

rand_rate_mean = [x[0] for x in rand_rates_by_coeff]
rand_rate_std = [x[1] for x in rand_rates_by_coeff]
rand_logit_mean = [x[0] for x in rand_logits_by_coeff]
rand_logit_std = [x[1] for x in rand_logits_by_coeff]

ax_a.plot(COEFFICIENTS, rand_rate_mean, color="#7f7f7f", linestyle="--", marker="s",
          markersize=3, label=f"Random mean (n={N_RANDOM_VECTORS})")
ax_a.fill_between(COEFFICIENTS,
                   [m - s for m, s in zip(rand_rate_mean, rand_rate_std)],
                   [m + s for m, s in zip(rand_rate_mean, rand_rate_std)],
                   color="#7f7f7f", alpha=0.15)
ax_b.plot(COEFFICIENTS, rand_logit_mean, color="#7f7f7f", linestyle="--", marker="s",
          markersize=3, label=f"Random mean (n={N_RANDOM_VECTORS})")
ax_b.fill_between(COEFFICIENTS,
                   [m - s for m, s in zip(rand_logit_mean, rand_logit_std)],
                   [m + s for m, s in zip(rand_logit_mean, rand_logit_std)],
                   color="#7f7f7f", alpha=0.15)

ax_a.axhline(baseline_rate * 100, ls=":", color="black", alpha=0.7, label="baseline")
ax_a.set_xlabel("Steering coefficient")
ax_a.set_ylabel("Sycophancy rate (%)")
ax_a.set_title(f"Binary sycophancy rate vs steering coefficient\n"
               f"(counterbalanced eval, {n_base} base x 2, {args.split} split)")
ax_a.legend(loc="best", fontsize=8, ncol=2)
ax_a.grid(True, alpha=0.3)

ax_b.axhline(baseline_logit, ls=":", color="black", alpha=0.7, label="baseline")
ax_b.axhline(0, ls="-", color="black", alpha=0.4, lw=0.7)
ax_b.set_xlabel("Steering coefficient")
ax_b.set_ylabel("Mean sycophancy logit  logp(syc) - logp(hon)")
ax_b.set_title("Continuous sycophancy preference vs steering coefficient")
ax_b.legend(loc="best", fontsize=8, ncol=2)
ax_b.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(f"{FIG_DIR}/fig1_steering_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("saved fig1_steering_curves.png")


# ---- Figure 2: axis projection vs sycophancy preference (baseline) ----
print("\n--- Figure 2 ---")
by_base = defaultdict(dict)
for r in baseline_rows:
    by_base[r["base_id"]][r["variant"]] = r

proj_per_base = []
logit_per_base = []
for b, vs in by_base.items():
    if "original" not in vs or "swapped" not in vs:
        continue
    proj_per_base.append(0.5 * (vs["original"]["axis_projection"] + vs["swapped"]["axis_projection"]))
    logit_per_base.append(0.5 * (vs["original"]["syc_logit"] + vs["swapped"]["syc_logit"]))
proj_per_base = np.array(proj_per_base)
logit_per_base = np.array(logit_per_base)

true_syc = 0; true_hon = 0; positional = 0
for b, vs in by_base.items():
    o = vs.get("original")
    s = vs.get("swapped")
    if o and s:
        if o["chose_sycophantic"] and s["chose_sycophantic"]:
            true_syc += 1
        elif (not o["chose_sycophantic"]) and (not s["chose_sycophantic"]):
            true_hon += 1
        else:
            positional += 1
print(f"[Baseline per-base classification]  true_syc={true_syc}  true_hon={true_hon}  positional={positional}")

q25 = float(np.percentile(logit_per_base, 25))
q75 = float(np.percentile(logit_per_base, 75))
low_mask = logit_per_base <= q25
high_mask = logit_per_base >= q75

fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

axA.scatter(proj_per_base, logit_per_base, s=14, alpha=0.55, color="#555555")
r_pearson_base, p_pearson_base = stats.pearsonr(proj_per_base, logit_per_base)
p_pearson_base = safe_pvalue(p_pearson_base)
m, c0 = np.polyfit(proj_per_base, logit_per_base, 1)
xs = np.linspace(proj_per_base.min(), proj_per_base.max(), 50)
axA.plot(xs, m * xs + c0, color="#d62728", lw=2,
         label=f"Pearson r = {r_pearson_base:+.3f} (p={fmt_p(p_pearson_base)})")
axA.axhline(0, ls="-", color="black", lw=0.5, alpha=0.5)
axA.set_xlabel("Mean Assistant Axis projection (per base question)")
axA.set_ylabel("Mean sycophancy logit (per base question)")
axA.set_title("Fig 2A: Axis projection vs sycophantic preference (baseline only)")
axA.legend()
axA.grid(True, alpha=0.3)

bins = np.linspace(proj_per_base.min() - 5, proj_per_base.max() + 5, 35)
axB.hist(proj_per_base[low_mask], bins=bins, alpha=0.6, color="#1f77b4",
         label=f"lowest-syc quartile (N={low_mask.sum()})")
axB.hist(proj_per_base[high_mask], bins=bins, alpha=0.6, color="#d62728",
         label=f"highest-syc quartile (N={high_mask.sum()})")
axB.set_xlabel("Mean Assistant Axis projection")
axB.set_ylabel("Number of base questions")
axB.set_title("Fig 2B: Projection distributions")
axB.legend()
axB.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(f"{FIG_DIR}/fig2_projection_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp else 0.0

t_welch, p_welch = stats.ttest_ind(proj_per_base[high_mask], proj_per_base[low_mask], equal_var=False)
p_welch = safe_pvalue(p_welch)
d = cohens_d(proj_per_base[high_mask], proj_per_base[low_mask])

proj_stats_baseline = {
    "N_base": len(proj_per_base),
    "pearson_r": float(r_pearson_base),
    "pearson_p": float(p_pearson_base),
    "mean_proj_highsyc_quartile": float(proj_per_base[high_mask].mean()),
    "mean_proj_lowsyc_quartile": float(proj_per_base[low_mask].mean()),
    "welch_t": float(t_welch),
    "welch_p": float(p_welch),
    "cohens_d": d,
    "per_base_binary": {"true_syc": true_syc, "true_hon": true_hon, "positional": positional},
}
print(f"  Baseline: Pearson r = {r_pearson_base:+.4f}  p = {fmt_p(p_pearson_base)}")
print(f"  Welch t = {t_welch:.3f}  p = {fmt_p(p_welch)}  Cohen's d = {d:.3f}")
print("saved fig2")


# ---- Pooled projection analysis across steered conditions ----
print("\n--- Pooled projection analysis ---")
# Collect post-steer projection and syc_logit across all (condition, coefficient)
# cells for real conditions. Post-steer projection is computed analytically.
cos_name_to_idx = {n: i for i, n in enumerate(cos["names"])}
cos_mat_full = np.array(cos["matrix"])
axis_idx_cos = cos_name_to_idx.get("assistant_axis", 0)

pooled_proj = []
pooled_logit = []
pooled_cond = []
for cond in CONDITIONS_REAL:
    cond_cos_idx = cos_name_to_idx.get(cond)
    if cond_cos_idx is None:
        continue
    cos_to_axis = float(cos_mat_full[axis_idx_cos, cond_cos_idx])
    for c in COEFFICIENTS:
        rows = idx[(cond, c)]
        if not rows:
            continue
        bl = base_level_logits(rows)
        base_projs = defaultdict(list)
        for r in rows:
            base_projs[r["base_id"]].append(r["axis_projection"] + c * cos_to_axis)
        for bid in bl:
            if bid in base_projs:
                pooled_proj.append(np.mean(base_projs[bid]))
                pooled_logit.append(bl[bid])
                pooled_cond.append(cond)

pooled_proj = np.array(pooled_proj)
pooled_logit = np.array(pooled_logit)
r_pooled, p_pooled = stats.pearsonr(pooled_proj, pooled_logit)
p_pooled = safe_pvalue(p_pooled)
print(f"  Pooled Pearson r (projection, syc_logit) = {r_pooled:+.4f}  p = {fmt_p(p_pooled)}")
print(f"  N data points = {len(pooled_proj)} (across {len(CONDITIONS_REAL)} conditions x {len(COEFFICIENTS)} coefficients)")

proj_stats_pooled = {
    "pearson_r": float(r_pooled),
    "pearson_p": float(p_pooled),
    "n_datapoints": len(pooled_proj),
    "note": "Pooled across all real conditions and coefficients, base-level averaged",
}


# ---- Response-token projection (captured during generation) vs prompt-last-token ----
# 02_evaluate_steering.py captures 'response_projection' = mean axis projection over
# 8 greedy-generated tokens per row. The changelog claim is that response-token
# projection is a better predictor of sycophancy than prompt-last-token. We test
# this by computing pooled Pearson r for both, on the same data points.
print("\n--- Response-token vs prompt-token projection ---")
pooled_resp_proj_bases = []
pooled_resp_proj = []
pooled_resp_logit = []
for cond in CONDITIONS_REAL:
    for c in COEFFICIENTS:
        rows = idx[(cond, c)]
        if not rows:
            continue
        bl = base_level_logits(rows)
        base_resp_projs = defaultdict(list)
        for r in rows:
            rp = r.get("response_projection")
            if rp is None or (isinstance(rp, float) and (rp != rp)):  # NaN check
                continue
            base_resp_projs[r["base_id"]].append(float(rp))
        for bid in bl:
            if bid in base_resp_projs and base_resp_projs[bid]:
                pooled_resp_proj.append(float(np.mean(base_resp_projs[bid])))
                pooled_resp_logit.append(bl[bid])
                pooled_resp_proj_bases.append((cond, c, bid))

if len(pooled_resp_proj) > 30:
    arr_rp = np.array(pooled_resp_proj)
    arr_rl = np.array(pooled_resp_logit)
    r_resp, p_resp = stats.pearsonr(arr_rp, arr_rl)
    p_resp = safe_pvalue(p_resp)
    print(f"  Response-token Pearson r (resp_projection, syc_logit) = {r_resp:+.4f}  "
          f"p = {fmt_p(p_resp)}  n = {len(arr_rp)}")
    print(f"  Prompt-token Pearson r (for comparison) = {r_pooled:+.4f}  n = {len(pooled_proj)}")
    proj_stats_response = {
        "pearson_r": float(r_resp),
        "pearson_p": float(p_resp),
        "n_datapoints": int(len(arr_rp)),
        "prompt_token_pearson_r_for_comparison": float(r_pooled),
        "note": "Pooled response-token axis projection vs base-level syc_logit. "
                "response_projection = mean axis-dot over 8 greedy response tokens.",
    }
else:
    print(f"  Not enough response projections ({len(pooled_resp_proj)}) to run the test")
    proj_stats_response = {
        "pearson_r": None, "pearson_p": None,
        "n_datapoints": int(len(pooled_resp_proj)),
        "prompt_token_pearson_r_for_comparison": float(r_pooled),
        "note": "response_projection missing or NaN for most rows",
    }


# ---- Figure 3: projection shift ----
print("\n--- Figure 3 ---")
cos_to_axis_full = {}
for n in cos["names"]:
    ci = cos_name_to_idx.get(n)
    if ci is not None:
        cos_to_axis_full[n] = float(cos_mat_full[axis_idx_cos, ci])

fig3, ax = plt.subplots(figsize=(9, 5.5))
for cond in ["assistant_axis", "skeptic", "contrarian", "random_0"]:
    if cond not in cos_to_axis_full:
        continue
    xs, ys, errs = [], [], []
    cos_ca = cos_to_axis_full[cond]
    for c in COEFFICIENTS:
        rows = idx[(cond, c)]
        if not rows:
            continue
        post_projs = [r["axis_projection"] + c * cos_ca for r in rows]
        xs.append(c)
        ys.append(float(np.mean(post_projs)))
        errs.append(float(np.std(post_projs) / math.sqrt(len(post_projs))))
    ax.errorbar(xs, ys, yerr=errs, label=f"{LABELS.get(cond, cond)}  (cos={cos_ca:+.2f})",
                color=COLORS.get(cond, "#7f7f7f"), marker="o", markersize=5, capsize=2,
                linestyle="--" if cond.startswith("random") else "-",
                lw=2.5 if cond == "assistant_axis" else 1.5)
ax.axhline(float(np.mean([r["axis_projection"] for r in baseline_rows])),
           ls=":", color="black", alpha=0.6, label="baseline projection")
ax.set_xlabel("Steering coefficient")
ax.set_ylabel("Post-steer Assistant Axis projection")
ax.set_title("Fig 3: How steering shifts the residual along the Assistant Axis")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(f"{FIG_DIR}/fig3_projection_shift.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("saved fig3")


# ---- Figure 4: best reduction per condition ----
print("\n--- Figure 4: best per condition ---")
# Best coefficient is selected in the direction the theory predicts:
#   "decrease" conditions -> coefficient that most REDUCES syc_logit vs random mean.
#   "increase" conditions -> coefficient that most INCREASES syc_logit vs random mean.
#   "unsigned" conditions -> coefficient with largest |deviation| (used for randoms).
# This keeps the conformist (increase-expected) best coefficient from being
# collapsed to the opposite sign by the old unidirectional selector.
#
# --locked-coefs-from: if provided, we skip selection entirely and read the
# per-condition coefficient from a prior tune-split run, then recompute the
# stats entry at that locked coefficient. This is how tune/test hygiene is
# enforced on reported numbers.


def _entry_at(cond, coef):
    """Build the 'best' entry dict for (cond, coef) from rates + random-mean."""
    k = f"{coef}"
    s = rates.get(cond, {}).get(k)
    if not s:
        return None
    rm = random_mean_at_coeff(k)
    rm = float(rm) if rm is not None else float(s["mean_syc_logit"])
    return {
        **s,
        "coefficient": coef,
        "excess_over_random_mean": rm - s["mean_syc_logit"],
        "random_mean_logit": rm,
    }


locked_coefs = None
if args.locked_coefs_from:
    with open(args.locked_coefs_from) as f:
        locked_coefs = json.load(f).get("best_coefs", {})
    print(f"Locked coefs loaded from {args.locked_coefs_from}: "
          f"{len(locked_coefs)} conditions")


best_per_cond = {}
for cond in CONDITIONS_REAL:
    if locked_coefs and cond in locked_coefs:
        best = _entry_at(cond, float(locked_coefs[cond]))
        best_per_cond[cond] = best
        if best:
            print(f"  locked for {cond}: coef={best['coefficient']:+}  "
                  f"logit={best['mean_syc_logit']:+.3f}  "
                  f"rate={best['binary_rate']*100:.1f}%  "
                  f"excess={best['excess_over_random_mean']:+.3f}")
        continue

    direction = EXPECTED_DIRECTION.get(cond, "decrease")
    best = None
    for c in COEFFICIENTS:
        if c == 0.0:
            continue
        entry = _entry_at(cond, c)
        if entry is None:
            continue
        excess = entry["excess_over_random_mean"]  # rm - cond_logit
        # "decrease": more-negative cond_logit -> larger positive excess.
        # "increase": more-positive cond_logit -> larger negative excess.
        # "unsigned": larger |excess| in either direction.
        if direction == "decrease":
            key = excess
        elif direction == "increase":
            key = -excess
        else:  # unsigned
            key = abs(excess)

        if best is None or key > best["_selector_key"]:
            best = {**entry, "_selector_key": key,
                    "selector_direction": direction}
    if best:
        best.pop("_selector_key", None)
    best_per_cond[cond] = best
    if best:
        print(f"  best for {cond} ({direction}): coef={best['coefficient']:+}  "
              f"logit={best['mean_syc_logit']:+.3f}  "
              f"rate={best['binary_rate']*100:.1f}%  "
              f"excess={best['excess_over_random_mean']:+.3f}")

# Persist selected coefs so a subsequent --split test run can --locked-coefs-from
# this file. Only write on tune (or when not locked) to avoid clobbering the
# tune-side selection with test-side numbers.
if locked_coefs is None:
    best_coefs_out = {
        "source_split": args.split,
        "best_coefs": {
            c: best_per_cond[c]["coefficient"]
            for c in best_per_cond if best_per_cond[c] is not None
        },
        "selector_direction": {
            c: EXPECTED_DIRECTION.get(c, "decrease") for c in best_per_cond
        },
    }
    with open(f"{RES_DIR}/best_coefs{suffix}.json", "w") as f:
        json.dump(best_coefs_out, f, indent=2)
    print(f"  wrote {RES_DIR}/best_coefs{suffix}.json")
else:
    # On locked runs, echo the locked file into a test-suffixed copy for
    # provenance (so the paper scripts only need to read best_coefs_test.json).
    with open(f"{RES_DIR}/best_coefs{suffix}.json", "w") as f:
        json.dump({
            "source_split": args.split,
            "locked_from": args.locked_coefs_from,
            "best_coefs": {c: best_per_cond[c]["coefficient"]
                           for c in best_per_cond if best_per_cond[c]},
            "selector_direction": {c: EXPECTED_DIRECTION.get(c, "decrease")
                                   for c in best_per_cond},
        }, f, indent=2)
    print(f"  wrote {RES_DIR}/best_coefs{suffix}.json (locked provenance)")

# Random mean "best" for display (representative)
rm_best = None
for c in COEFFICIENTS:
    if c == 0.0:
        continue
    k = f"{c}"
    rm = random_mean_at_coeff(k)
    if rm is not None and (rm_best is None or rm < rm_best["mean_syc_logit"]):
        rm_best = {"mean_syc_logit": rm, "coefficient": c, "binary_rate": 0.5}

fig4, ax = plt.subplots(figsize=(10, 5.5))
plot_conds = [c for c in CONDITIONS_REAL if best_per_cond.get(c)]
names_fig4 = [LABELS[c] for c in plot_conds]
logits_fig4 = [best_per_cond[c]["mean_syc_logit"] for c in plot_conds]
coefs_fig4 = [best_per_cond[c]["coefficient"] for c in plot_conds]

errs_lo, errs_hi = [], []
for c in plot_conds:
    rows = idx[(c, best_per_cond[c]["coefficient"])]
    if rows:
        lo, hi = bootstrap_ci(rows, logit_of, n=500, seed=1)
        errs_lo.append(best_per_cond[c]["mean_syc_logit"] - lo)
        errs_hi.append(hi - best_per_cond[c]["mean_syc_logit"])
    else:
        errs_lo.append(0)
        errs_hi.append(0)

x = np.arange(len(plot_conds))
bar_colors = [COLORS[c] for c in plot_conds]
bars = ax.bar(x, logits_fig4, yerr=[errs_lo, errs_hi], color=bar_colors, capsize=4,
              edgecolor="black", linewidth=0.8)
ax.axhline(baseline_logit, ls=":", color="black", lw=1.2, label=f"baseline = {baseline_logit:+.2f}")
ax.axhline(0, ls="-", color="black", lw=0.7, alpha=0.5)
for bar, coef in zip(bars, coefs_fig4):
    ax.annotate(f"c={coef:+.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(names_fig4, rotation=20, ha="right")
ax.set_ylabel("Mean sycophancy logit (lower = less sycophantic)")
ax.set_title(f"Fig 4: Best sycophancy reduction per condition (bootstrap 95% CI)\n[{args.split} split]")
ax.legend(loc="best")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig4.savefig(f"{FIG_DIR}/fig4_best_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("saved fig4")


# ---- Figure 5: cosine similarity heatmap ----
print("\n--- Figure 5 ---")
# Show real conditions + random_0 only for readability
show_names_cos = [n for n in cos["names"] if not n.startswith("random_")] + \
                 ([n for n in cos["names"] if n == "random_0"] if "random_0" in cos["names"] else [])
show_idxs_cos = [cos["names"].index(n) for n in show_names_cos if n in cos["names"]]
mat_show = cos_mat_full[np.ix_(show_idxs_cos, show_idxs_cos)]

fig5, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mat_show, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(show_idxs_cos)))
ax.set_yticks(range(len(show_idxs_cos)))
show_labels = [LABELS.get(show_names_cos[i], show_names_cos[i]) for i in range(len(show_idxs_cos))]
ax.set_xticklabels(show_labels, rotation=35, ha="right")
ax.set_yticklabels(show_labels)
for i in range(len(show_idxs_cos)):
    for j in range(len(show_idxs_cos)):
        color = "white" if abs(mat_show[i, j]) > 0.55 else "black"
        ax.text(j, i, f"{mat_show[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, label="cosine similarity")
ax.set_title(f"Fig 5: Cosine similarity between steering vectors (L{TARGET_LAYER})")
plt.tight_layout()
fig5.savefig(f"{FIG_DIR}/fig5_vector_similarities.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("saved fig5")


# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n--- Statistical tests ---")
tests = {}

# ---- 1. Primary test: paired Wilcoxon on base-level syc_logit ----
# For each condition, compare base-level syc_logit at best coefficient vs baseline.
tests["primary_wilcoxon_best_vs_baseline"] = {}
baseline_bl = base_level_logits(baseline_rows)
base_ids_sorted = sorted(baseline_bl.keys())

raw_pvals_wilcoxon = []
wilcoxon_entries = []
for cond in CONDITIONS_REAL:
    bc = best_per_cond.get(cond, {})
    if not bc:
        continue
    best_c = bc["coefficient"]
    best_rows = idx[(cond, best_c)]
    if not best_rows:
        continue
    best_bl = base_level_logits(best_rows)
    # Paired comparison: same base questions
    common_bases = sorted(set(baseline_bl.keys()) & set(best_bl.keys()))
    baseline_vals = np.array([baseline_bl[b] for b in common_bases])
    best_vals = np.array([best_bl[b] for b in common_bases])
    diffs = best_vals - baseline_vals  # steered - baseline

    # Direction-aware Wilcoxon. For "decrease" conditions we run a one-sided
    # test asking whether diffs < 0 (steering lowers syc_logit); for "increase"
    # conditions we ask whether diffs > 0. For unsigned (random controls) we
    # keep the two-sided test. A two-sided version is always also reported.
    direction = EXPECTED_DIRECTION.get(cond, "decrease")
    if np.all(diffs == 0):
        stat_two, pval_two = 0.0, 1.0
        stat_dir, pval_dir = 0.0, 1.0
    else:
        stat_two, pval_two = stats.wilcoxon(diffs, alternative="two-sided")
        if direction == "decrease":
            stat_dir, pval_dir = stats.wilcoxon(diffs, alternative="less")
        elif direction == "increase":
            stat_dir, pval_dir = stats.wilcoxon(diffs, alternative="greater")
        else:
            stat_dir, pval_dir = stat_two, pval_two
    pval_two = safe_pvalue(pval_two)
    pval_dir = safe_pvalue(pval_dir)

    # The primary p-value is the one-sided test in the expected direction
    # (for directionally-signed conditions) or the two-sided test (for
    # unsigned conditions). Holm correction is applied across this primary
    # family below. The two-sided p is kept as a secondary column.
    pval = pval_dir
    stat = stat_dir

    raw_pvals_wilcoxon.append(pval)
    wilcoxon_entries.append({
        "condition": cond,
        "best_coef": best_c,
        "n_base": len(common_bases),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "statistic": float(stat),
        "p_value_raw": float(pval),
        "p_value_two_sided_raw": float(pval_two),
        "alternative": ("less" if direction == "decrease"
                        else "greater" if direction == "increase"
                        else "two-sided"),
        "expected_direction": direction,
        "best_logit": bc["mean_syc_logit"],
        "best_rate": bc["binary_rate"],
    })

# Apply multiple comparison correction
reject_w, adj_w = apply_mcc(raw_pvals_wilcoxon)
for i, entry in enumerate(wilcoxon_entries):
    entry["p_value_adjusted"] = float(adj_w[i])
    entry["significant_after_mcc"] = bool(reject_w[i])
    tests["primary_wilcoxon_best_vs_baseline"][entry["condition"]] = entry
    print(f"  Wilcoxon {entry['condition']}: diff={entry['mean_diff']:+.3f}  "
          f"p_raw={fmt_p(entry['p_value_raw'])}  p_adj={fmt_p(entry['p_value_adjusted'])}  "
          f"{'*' if entry['significant_after_mcc'] else 'ns'}")


# ---- 2. McNemar tests (secondary, with exact when needed) ----
def paired_mcnemar(rows_a, rows_b):
    """Paired McNemar with exact test when off-diagonal cells < 5."""
    base_map = {r["question_id"]: r["chose_sycophantic"] for r in rows_b}
    b01 = b10 = b00 = b11 = 0
    for r in rows_a:
        qid = r["question_id"]
        if qid not in base_map:
            continue
        a = base_map[qid]
        b = r["chose_sycophantic"]
        if a and not b:
            b10 += 1
        elif (not a) and b:
            b01 += 1
        elif a and b:
            b11 += 1
        else:
            b00 += 1
    tbl = [[b11, b10], [b01, b00]]
    # Use exact test when off-diagonal cells are small
    use_exact = min(b01, b10) < 5
    res = mcnemar(tbl, exact=use_exact, correction=not use_exact)
    return {"b11": b11, "b10": b10, "b01": b01, "b00": b00,
            "exact": use_exact,
            "statistic": float(res.statistic), "p_value": safe_pvalue(float(res.pvalue))}

tests["mcnemar_best_vs_baseline"] = {}
raw_pvals_mcnemar = []
mcnemar_entries = []
for cond in CONDITIONS_REAL:
    bc = best_per_cond.get(cond)
    if not bc:
        continue
    best_c = bc["coefficient"]
    rows_best = idx[(cond, best_c)]
    mc = paired_mcnemar(rows_best, baseline_rows)
    raw_pvals_mcnemar.append(mc["p_value"])
    mcnemar_entries.append((cond, mc, best_c))

reject_mc, adj_mc = apply_mcc(raw_pvals_mcnemar)
for i, (cond, mc, best_c) in enumerate(mcnemar_entries):
    mc["p_value_adjusted"] = float(adj_mc[i])
    mc["significant_after_mcc"] = bool(reject_mc[i])
    mc["best_coef"] = best_c
    mc["exact_test_used"] = mc.pop("exact")
    tests["mcnemar_best_vs_baseline"][cond] = mc

# ---- 3. Real vs random-mean at each condition's best coefficient ----
tests["real_vs_random_mean"] = {}
for cond in CONDITIONS_REAL:
    bc = best_per_cond.get(cond)
    if not bc:
        continue
    best_c = bc["coefficient"]
    k = f"{best_c}"
    real_logit = bc["mean_syc_logit"]
    rm = random_mean_at_coeff(k)
    # Collect all random rows at this coefficient for a proper test
    rand_logits_at_c = []
    for rc in CONDITIONS_RANDOM:
        s = rates.get(rc, {}).get(k)
        if s:
            rand_logits_at_c.append(s["mean_syc_logit"])
    tests["real_vs_random_mean"][cond] = {
        "coef": best_c,
        "real_logit": real_logit,
        "random_mean_logit": float(rm) if rm is not None else None,
        "random_std_logit": float(np.std(rand_logits_at_c)) if len(rand_logits_at_c) > 1 else None,
        "n_random_vectors": len(rand_logits_at_c),
        "excess_over_random_mean": float(rm - real_logit) if rm is not None else None,
    }


# ---- 4. Dose-response: ROW-LEVEL Spearman (proper, not 6-point aggregate) ----
print("\n--- Dose-response (row-level Spearman) ---")
tests["dose_response_row_level"] = {}
pos_coeffs = [c for c in COEFFICIENTS if c > 0]
for cond in CONDITIONS_REAL:
    all_coeffs_row = []
    all_logits_row = []
    for c in pos_coeffs:
        rows = idx[(cond, c)]
        for r in rows:
            all_coeffs_row.append(c)
            all_logits_row.append(r["syc_logit"])
    if len(all_coeffs_row) < 10:
        continue
    rho, pval = stats.spearmanr(all_coeffs_row, all_logits_row)
    pval = safe_pvalue(pval)
    tests["dose_response_row_level"][cond] = {
        "rho": float(rho),
        "p_value": float(pval),
        "n_rows": len(all_coeffs_row),
        "n_coefficients": len(pos_coeffs),
        "note": "Row-level Spearman across all positive coefficients (not 6-point aggregate)",
    }
    print(f"  {cond}: rho={rho:+.4f}  p={fmt_p(pval)}  n={len(all_coeffs_row)}")

# Also keep the aggregate version for comparison, but label it clearly
tests["dose_response_aggregate_6pt"] = {}
for cond in CONDITIONS_REAL + CONDITIONS_RANDOM[:1]:
    logits_agg = []
    for c in pos_coeffs:
        k = f"{c}"
        s = rates.get(cond, {}).get(k)
        if s:
            logits_agg.append(s["mean_syc_logit"])
    if len(logits_agg) == len(pos_coeffs):
        rho, pval = stats.spearmanr(pos_coeffs, logits_agg)
        pval = safe_pvalue(pval)
        tests["dose_response_aggregate_6pt"][cond] = {
            "rho": float(rho),
            "p_value": float(pval),
            "n_points": len(pos_coeffs),
            "note": "6-point aggregate Spearman (included for comparison only; see row-level for primary)",
        }


# ---- 5. Axis projection analysis ----
tests["axis_projection_baseline"] = proj_stats_baseline
tests["axis_projection_pooled"] = proj_stats_pooled
tests["axis_projection_response_token"] = proj_stats_response


# ---- 6. CAA decomposition analysis ----
decomp_path = f"{ROOT}/vectors/steering/caa_decomposition.json"
if os.path.exists(decomp_path):
    print("\n--- CAA Decomposition Analysis ---")
    with open(decomp_path) as f:
        decomp = json.load(f)

    tests["caa_decomposition"] = {}
    for cond_name, d in decomp.items():
        tests["caa_decomposition"][cond_name] = {
            "cosine_with_caa": d["cosine_with_caa"],
            "caa_component_norm": d["caa_component_norm"],
            "residual_norm": d["residual_norm"],
        }
        # Check if decomposed vectors were evaluated
        comp_key = f"{cond_name}_caa_component"
        resid_key = f"{cond_name}_residual"
        for key_label, key_name in [("caa_component", comp_key), ("residual", resid_key)]:
            # Look for results from ablation run (if user ran 02 with component vectors)
            best_c_full = best_per_cond.get(cond_name, {}).get("coefficient")
            if best_c_full:
                rows_comp = idx.get((key_name, best_c_full), [])
                if rows_comp:
                    tests["caa_decomposition"][cond_name][f"{key_label}_logit"] = logit_of(rows_comp)
                    tests["caa_decomposition"][cond_name][f"{key_label}_rate"] = rate_of(rows_comp)

        print(f"  {cond_name}: cos(CAA)={d['cosine_with_caa']:+.4f}")

    # Efficiency analysis: sycophancy_reduction / |coefficient| vs cosine with CAA
    print("\n--- Efficiency Analysis ---")
    tests["efficiency"] = {}
    for cond in CONDITIONS_REAL:
        bc = best_per_cond.get(cond)
        if not bc or bc["coefficient"] == 0:
            continue
        reduction = baseline_logit - bc["mean_syc_logit"]
        magnitude = abs(bc["coefficient"])
        efficiency = reduction / magnitude if magnitude > 0 else 0
        cos_caa = decomp.get(cond, {}).get("cosine_with_caa", 0.0) if cond in decomp else 0.0
        tests["efficiency"][cond] = {
            "reduction": float(reduction),
            "magnitude": magnitude,
            "efficiency": float(efficiency),
            "cosine_with_caa": float(cos_caa),
        }
        print(f"  {cond}: reduction={reduction:+.3f}  efficiency={efficiency:.6f}  cos(CAA)={cos_caa:+.4f}")
else:
    print("\n--- CAA decomposition not found (run 02b_extract_caa.py + 01) ---")


# ---- 7. PCA persona-space visualization ----
print("\n--- Figure 6: Persona Space PCA ---")
try:
    from sklearn.decomposition import PCA

    # Collect all real steering vectors for PCA
    vec_names_pca = []
    vec_data_pca = []
    for name in cos["names"]:
        if name.startswith("random_"):
            if name != "random_0":
                continue  # only include one random for reference
        vec_path = f"{ROOT}/vectors/steering/{name}_unit.pt"
        if os.path.exists(vec_path):
            v = torch.load(vec_path, map_location="cpu", weights_only=False).float().numpy()
            vec_names_pca.append(name)
            vec_data_pca.append(v)

    if len(vec_data_pca) >= 3:
        vec_matrix = np.stack(vec_data_pca)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vec_matrix)

        fig6, ax = plt.subplots(figsize=(10, 8))
        for i, name in enumerate(vec_names_pca):
            color = COLORS.get(name, "#333333")
            label = LABELS.get(name, name)
            marker = "o"
            size = 100
            if name in CONFORMIST_ROLES:
                marker = "s"  # square for conformist
            elif name.startswith("random"):
                marker = "x"
                size = 80
            elif name == "caa":
                marker = "D"  # diamond for CAA
                size = 120

            # Annotate with sycophancy reduction if available
            bc = best_per_cond.get(name)
            annot = label
            if bc:
                annot += f"\n(logit {bc['mean_syc_logit']:+.2f})"

            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=size, marker=marker,
                       zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(annot, (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=color)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.set_title("Fig 6: Persona Space (PCA of steering vectors)\n"
                      "o = critical roles, s = conformist roles, D = CAA, x = random")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig6.savefig(f"{FIG_DIR}/fig6_persona_space_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig6)
        print("saved fig6_persona_space_pca.png")

        tests["pca_persona_space"] = {
            "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pc2_variance_ratio": float(pca.explained_variance_ratio_[1]),
            "vector_names": vec_names_pca,
            "coordinates": coords.tolist(),
        }
    else:
        print("  Not enough vectors for PCA (need >= 3)")
except ImportError:
    print("  sklearn not available, skipping PCA")


# ---- Save tests ----
with open(f"{RES_DIR}/statistical_tests{suffix}.json", "w") as f:
    json.dump(tests, f, indent=2, default=str)


# ============================================================
# SUMMARY
# ============================================================
print("\n--- Summary ---")

# "Most effective reducer" should only consider conditions whose expected
# direction is decrease and whose best-coef cell is not flagged degraded.
def _is_degraded_at_best(cond):
    b = best_per_cond.get(cond)
    if not b:
        return False
    key = f"{b['coefficient']}"
    return bool(degraded.get(cond, {}).get(key, False))

reducer_candidates = [
    c for c in CONDITIONS_REAL
    if best_per_cond.get(c)
    and EXPECTED_DIRECTION.get(c, "decrease") == "decrease"
    and not _is_degraded_at_best(c)
]
if reducer_candidates:
    most_effective = min(reducer_candidates,
                         key=lambda c: best_per_cond[c]["mean_syc_logit"])
else:
    most_effective = None

summary_lines = [
    f"=== RESULTS SUMMARY ({args.split} split) ===",
    "",
    f"Model: {CONDITIONS_REAL[0] if CONDITIONS_REAL else 'N/A'}  (target layer {TARGET_LAYER})",
    f"Eval: philpapers2020, N={n_base} base questions x 2 orderings = {len(baseline_rows)} rows",
    f"Random controls: {N_RANDOM_VECTORS} independent random unit vectors",
    f"Multiple comparison correction: {MCC_METHOD}",
    f"Locked coefs: {'yes -- from ' + args.locked_coefs_from if locked_coefs else 'no (selected on this split)'}",
    "",
    f"Baseline binary sycophancy rate: {baseline_rate*100:.1f}%",
    f"Baseline mean sycophancy logit:  {baseline_logit:+.3f}",
    "",
    "--- Best coefficient per condition (direction-aware selector) ---",
    "(coef selected to maximise expected-direction effect vs random MEAN;"
    "  conformist roles use increase selector, all others use decrease or unsigned)",
]

ordered = sorted(
    [c for c in CONDITIONS_REAL if best_per_cond.get(c)],
    key=lambda c: best_per_cond[c]["mean_syc_logit"],
)
for cond in ordered:
    b = best_per_cond[cond]
    w = tests["primary_wilcoxon_best_vs_baseline"].get(cond, {})
    delta_logit = b["mean_syc_logit"] - baseline_logit
    delta_rate_pp = (b["binary_rate"] - baseline_rate) * 100
    direction = EXPECTED_DIRECTION.get(cond, "decrease")
    sig_str = ""
    if w:
        sig = "*" if w.get("significant_after_mcc") else "ns"
        alt = w.get("alternative", "two-sided")
        sig_str = (f"  Wilcoxon[{alt}] p_adj={fmt_p(w.get('p_value_adjusted', 1.0))} "
                   f"[{sig}]")
    deg_str = "  DEGRADED" if _is_degraded_at_best(cond) else ""
    summary_lines.append(
        f"  {LABELS[cond]:<18} [{direction:<9}] coef={b['coefficient']:+6.0f}  "
        f"logit={b['mean_syc_logit']:+.3f} (d{delta_logit:+.3f})  "
        f"rate={b['binary_rate']*100:5.1f}% (d{delta_rate_pp:+.1f}pp)"
        f"{sig_str}{deg_str}"
    )

summary_lines += [
    "",
    "--- Axis projection analysis ---",
    f"  Baseline (prompt-last-token): Pearson r = {proj_stats_baseline['pearson_r']:+.4f}  "
    f"p = {fmt_p(proj_stats_baseline['pearson_p'])}",
    f"  Pooled (all conditions, prompt-last-token, analytic): Pearson r = "
    f"{proj_stats_pooled['pearson_r']:+.4f}  p = {fmt_p(proj_stats_pooled['pearson_p'])}  "
    f"(n={proj_stats_pooled['n_datapoints']})",
]
if proj_stats_response.get("pearson_r") is not None:
    summary_lines.append(
        f"  Pooled (all conditions, RESPONSE-token, measured): Pearson r = "
        f"{proj_stats_response['pearson_r']:+.4f}  p = "
        f"{fmt_p(proj_stats_response['pearson_p'])}  (n={proj_stats_response['n_datapoints']})"
    )
summary_lines += [
    "",
    "--- Dose-response (row-level Spearman, positive coefficients) ---",
]
for cond in CONDITIONS_REAL:
    d = tests["dose_response_row_level"].get(cond, {})
    if d:
        summary_lines.append(
            f"  {LABELS[cond]:<18}  rho = {d['rho']:+.4f}  p = {fmt_p(d['p_value'])}  (n={d['n_rows']})"
        )

summary_lines += [
    "",
    "--- Key findings ---",
]
if most_effective is not None:
    summary_lines.append(
        f"1. Most effective reducer (non-degraded, decrease-expected): "
        f"{LABELS[most_effective]} "
        f"(logit {best_per_cond[most_effective]['mean_syc_logit']:+.3f}, "
        f"coef {best_per_cond[most_effective]['coefficient']:+.0f})"
    )
else:
    summary_lines.append(
        "1. No non-degraded decrease-expected condition produced a reduction."
    )

# Conformist bidirectionality summary
conformist_entries = [
    (c, best_per_cond[c], tests["primary_wilcoxon_best_vs_baseline"].get(c, {}))
    for c in CONFORMIST_ROLES if best_per_cond.get(c)
]
if conformist_entries:
    n_sig_increase = sum(
        1 for _, b, w in conformist_entries
        if b["mean_syc_logit"] > baseline_logit and w.get("significant_after_mcc")
    )
    summary_lines.append(
        f"1b. Bidirectionality check: {n_sig_increase}/{len(conformist_entries)} "
        f"conformist roles show a statistically significant INCREASE in "
        f"sycophancy at their best (increase-selector) coefficient."
    )

# Check if axis is significant
ax_w = tests["primary_wilcoxon_best_vs_baseline"].get("assistant_axis", {})
if ax_w.get("significant_after_mcc"):
    summary_lines.append(
        f"2. Assistant Axis significantly reduces sycophancy "
        f"(Wilcoxon p_adj = {fmt_p(ax_w['p_value_adjusted'])})"
    )
else:
    summary_lines.append(
        f"2. Assistant Axis does NOT significantly reduce sycophancy after MCC "
        f"(Wilcoxon p_adj = {fmt_p(ax_w.get('p_value_adjusted', 1.0))})"
    )

if proj_stats_pooled["pearson_p"] < 0.05:
    summary_lines.append(
        f"3. Pooled projection analysis: axis projection IS correlated with sycophancy "
        f"(r = {proj_stats_pooled['pearson_r']:+.3f}, p = {fmt_p(proj_stats_pooled['pearson_p'])})"
    )
else:
    summary_lines.append(
        f"3. Pooled projection analysis: axis projection does NOT predict sycophancy "
        f"(r = {proj_stats_pooled['pearson_r']:+.3f}, p = {fmt_p(proj_stats_pooled['pearson_p'])})"
    )

summary_lines += [
    "",
    "Full stats in results/statistical_tests.json.  Figures in figures/.",
]

summary_text = "\n".join(summary_lines)
print()
print(summary_text)
with open(f"{RES_DIR}/summary{suffix}.txt", "w") as f:
    f.write(summary_text + "\n")

print(f"\nScript 3 done ({args.split} split).")
