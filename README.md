# General Persona Directions Reduce Sycophancy via Activation Steering

We show that general-purpose persona directions — extracted from open-ended
role-playing, not from sycophancy data — reduce sycophantic behavior in
**Gemma 2 27B Instruct** and **Qwen 3 32B**, and we decompose why this
works geometrically and mechanistically.

This is the **public, reproducible** mirror of the project. The code is
organized as two parallel pipelines, one per model:

- [`gemma/`](./gemma/) — Gemma 2 27B Instruct pipeline.
  `TARGET_LAYER = 22` of 46. Coefficient sweep in the `±5000` range.
  Setup, running instructions, and per-step checkpoints are in
  [`gemma/README.md`](./gemma/README.md) (655 lines, the original
  step-by-step Lambda guide). The 10-minute copy-paste version is in
  [`gemma/LAMBDA_QUICKSTART.md`](./gemma/LAMBDA_QUICKSTART.md).
- [`qwen/`](./qwen/) — Qwen 3 32B pipeline.
  `TARGET_LAYER = 32` of 64. Coefficient sweep rescaled `~10×` smaller
  because of Qwen's smaller layer-32 activation norms (~408 vs Gemma's
  ~19,750 mean per-token L2 norm). Setup notes, including the
  vector-regeneration recipe, are in [`qwen/README.md`](./qwen/README.md).
  The Qwen vector files are gitignored on this side because the upstream
  release for `qwen-3-32b/` (`lu-christina/assistant-axis-vectors`) is
  not ours to redistribute.
- [`paper/`](./paper/) — ICML 2026 figures for the cross-model writeup.

Aggregated cross-model results (Δlogit tables, family means, cosine
heatmaps, qualitative samples) live in a separate repo:
**`kelkarI/sycophancy-clean-results`**. This repo is the reproducible
**source** of those numbers; the aggregator pulls from
`gemma/results/` and `qwen/results/` directly.

---

## What this experiment does

**The question.** Can general persona vectors (skeptic, judge, devil's
advocate, etc.) reduce sycophancy as effectively as a targeted
sycophancy-specific vector (CAA, Rimsky et al. 2024)? And if so, how
does each vector reduce it — through healthy disagreement, neutral
hedging, or model degradation?

**The answer (test split, Holm-corrected, 3 seeds per model).**

| | Gemma 2 27B | Qwen 3 32B |
|---|---:|---:|
| Baseline syc-logit (no steering) | +1.009 | +3.024 |
| CAA (targeted) Δlogit | −0.874 | −1.989 |
| Skeptic Δlogit | **−0.706** (81% of CAA) | **−1.847** (93% of CAA) |
| Devil's Advocate Δlogit | −0.516 (59%) | −2.296 (115%) |
| Judge Δlogit | −0.551 (63%) | −1.723 (87%) |
| Random null (n=10 vectors × 8 coefs aggregate) | −0.254 | −1.058 |

Critical-role vectors are Holm-significant on all 3 test seeds on both
models. Conformist roles are heterogeneous (some reduce, some increase,
some are degraded). All role–CAA cosines satisfy `|cos| < 0.17` at the
injection layer; the *sign* of the cosine flips between Gemma and Qwen,
which we make explicit as a caveat on mechanistic-independence claims.

Full numbers: [`paper/tables/main_results_both_models.md`](./paper/tables/main_results_both_models.md).

---

## How to reproduce

The two pipelines are independent. Pick a model and follow the matching
quickstart:

**Gemma 2 27B** (~10 hours, ~$25 on H100, full multi-seed run):
```bash
git clone https://github.com/vkmk1/Sycophancy-Steering.git
cd Sycophancy-Steering/gemma/scripts
# follow gemma/LAMBDA_QUICKSTART.md or gemma/README.md
```

**Qwen 3 32B** (~similar wall, slightly higher RAM ceiling at 32B):
```bash
git clone https://github.com/vkmk1/Sycophancy-Steering.git
cd Sycophancy-Steering/qwen/scripts
# follow qwen/README.md (note the vector-regeneration recipe at the top)
```

Both pipelines need:
- 1× H100 80GB or A100 80GB
- HuggingFace access to `google/gemma-2-27b-it` (Gemma) or
  `Qwen/Qwen3-32B` (Qwen) — both are gated, request access on the
  model page first
- Clone of [`safety-research/assistant-axis`](https://github.com/safety-research/assistant-axis) at `/home/ubuntu/assistant-axis`
  (provides the `ActivationSteering` context manager)

---

## Repo layout

```
.
├── README.md                       (this file)
├── .gitignore
├── gemma/                          Gemma 2 27B pipeline
│   ├── README.md                   per-pipeline README (operational details)
│   ├── LAMBDA_QUICKSTART.md        copy-paste Lambda recipe
│   ├── GAP_ANALYSIS.md             pre-publication gap audit
│   ├── PIPELINE_MAP.md             dataflow diagram across the 11 scripts
│   ├── scripts/                    20 scripts (setup → eval → analysis)
│   │   ├── config.py               centralized constants
│   │   ├── 00_setup.py             [step 1] download + sanity check
│   │   ├── 00b_rebuild_eval.py     [step 3] counterbalanced 600-row eval set
│   │   ├── 01_prepare_steering_vectors.py   [step 5] vectors + decomposition
│   │   ├── 02_evaluate_steering.py [step 7] main eval (--split, --resume)
│   │   ├── 02b_extract_caa.py      [step 4] CAA extraction (Rimsky et al.)
│   │   ├── 02c_evaluate_decomposition.py    CAA-component vs residual eval
│   │   ├── 02d_response_capture.py response-token axis projection
│   │   ├── 03_analysis.py          [step 8] figures + stats + PCA
│   │   ├── over_correction_check.py [step 9] structured ground-truth probes
│   │   ├── qual_check.py           [step 10] free-form generation comparison
│   │   ├── qual_check_caa.py       qual_check, CAA-only
│   │   ├── qual_check_conformist.py qual_check, conformist roles
│   │   ├── reconstruct_checkpoints.py recover from partial runs
│   │   ├── residual_standalone_report.py residual-only reporting
│   │   ├── fetch_external.py       one-shot external dep downloader
│   │   ├── pilot_debug.py          verify hooks + sign convention
│   │   ├── pilot_scale.py          find coefficient regime
│   │   ├── find_skeptic_agreement.py
│   │   └── run_multiseed.py        multi-seed runner
│   ├── data/                       eval datasets (eval_data.json + sycophancy/)
│   ├── results/                    summaries + per-seed JSONs
│   ├── figures/                    fig1–fig5
│   └── vectors/                    7 unit-normed steering vectors
├── qwen/                           Qwen 3 32B pipeline (parallels gemma/)
│   ├── README.md
│   ├── .gitignore
│   └── scripts/                    23 scripts (gemma's 20 + 3 Qwen-specific:
│                                    aggregate_multiseed.py,
│                                    build_vectors_from_official.py,
│                                    extract_all_vectors.py)
└── paper/                          ICML 2026 submission package
    ├── fig1_deltas.pdf
    ├── fig2_cosines.pdf
    ├── fig3_curves_family.pdf
    ├── fig4_per_seed.pdf
    ├── fig5_per_condition.pdf
    ├── figures/                    extended figure set (fig1–fig8)
    ├── tables/                     CSV + Markdown of main_results_both_models
    └── *.sty / *.bst / *.tex       LaTeX styling helpers
```

---

## Key design decisions

These apply to **both** pipelines (Gemma and Qwen). The per-pipeline
READMEs go deeper on each.

**Counterbalancing.** Gemma 2 27B has a ~93% positional A-bias. Every
base question is evaluated in both A/B orderings; the A-bias cancels
in the mean. Qwen has the same property (84% A-bias) and uses the
same protocol.

**Tune/test split.** Best-coefficient selection and final reporting
use different data splits (at the base-question level, so
counterbalanced pairs stay together). This prevents selection bias —
the #1 issue in the original experiment design.

**10 random controls.** Per seed, 10 independently-drawn unit-Gaussian
random vectors are evaluated alongside the real conditions, sharing
the same coefficient sweep and degradation-flagging rules. The
"random null" reported in tables is the cross-vector × cross-coef
mean of these random Δlogits per seed, then aggregated across seeds.

**Multi-seed.** 3 test seeds (seed_42, seed_7, seed_123) plus 2
tune-only seeds (seed_456, seed_789) per model. Holm–Bonferroni
correction is applied across all conditions in the family (24 on
each model with the residual conditions included).

**Layer choice.** `TARGET_LAYER = 22` on Gemma 2 27B (mid-stack of 46),
`TARGET_LAYER = 32` on Qwen 3 32B (mid-stack of 64). Both are the
canonical mid-stack layers for the `assistant_axis` library.

**Cross-model coefficient rescale.** Gemma uses `±5000` range;
Qwen uses `±500`. The `~10×` rescale reflects Qwen's smaller activation
norm at layer 32 — though this scaling does *not* equalize the
fractional perturbation `ε = |α| / ‖h̄‖` across models (Qwen's
empirical ε at the locked α is ~5× Gemma's). Per-pipeline READMEs
discuss this.

---

## Citing

If you build on this code, please cite the paper (forthcoming, ICML 2026).
For the time being:

```
@misc{sycophancy-steering-2026,
  title  = {General Persona Directions Reduce Sycophancy via Activation Steering},
  author = {{Anonymous}},
  year   = {2026},
  note   = {ICML 2026 submission. Code: \url{https://github.com/vkmk1/Sycophancy-Steering}},
}
```
