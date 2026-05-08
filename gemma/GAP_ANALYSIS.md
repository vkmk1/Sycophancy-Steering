# GAP_ANALYSIS.md

Ruthless audit of what the current code supports vs what a conference paper needs, organised around the three research objectives. Claims without file/line citations are flagged.

Key terminology: *critical roles* = `{devils_advocate, contrarian, skeptic, judge, scientist}`; *conformist roles* = `{servant, diplomat, disciple, peacemaker}`; *real conditions* = `CONDITIONS_REAL` from `config.py` = assistant_axis + 9 roles + caa (11 total); *random conditions* = `random_0..random_9`.

---

## Objective 1 — Role vectors reduce sycophancy; conformist roles increase it (bidirectionality)

### What the code does

- Evaluates all 9 roles at 13 coefficients × 600 counterbalanced rows in `02_evaluate_steering.py`.
- Primary test in `03_analysis.py` L488–536: paired Wilcoxon signed-rank on **base-level mean `syc_logit`**, best coef vs baseline, Holm-Bonferroni across all 11 real conditions.
- Secondary: McNemar with exact test when `min(b01,b10)<5` (L540–585); row-level Spearman dose-response on positive coefficients (L614–636); real-vs-random-mean logit delta at best coef (L587–610).
- Random controls averaged into a ±std band in `fig1` (L154–178).
- Counterbalancing (A/B swap with flipped syc label) is done in `00b_rebuild_eval.py`; base-level aggregation in `03_analysis.py` `base_level_logits()` (L107) respects pairing.

### What is missing or broken

1. **Conformist roles are evaluated with the wrong "best coefficient" selector.** `03_analysis.py` L382–399 picks `best_per_cond[cond]` as `argmax(random_mean - cond_logit)`, i.e. the coef that **reduces** sycophancy most vs the random baseline. For a conformist role whose true effect is an **increase** in sycophancy, this picks the opposite-sign or smallest-increase coef — the one that looks most like random — and then the Wilcoxon in L488–536 tests a non-effect. The conformist entries in `results/statistical_tests.json → primary_wilcoxon_best_vs_baseline` are therefore meaningless as a bidirectionality test. This is the single largest Objective-1 gap.
   - **Fix:** introduce an expected-direction tag per condition (`decrease` for axis/critical/caa, `increase` for conformist). Select best coef as `argmin(cond_logit - random_mean)` for `decrease` and `argmax(cond_logit - random_mean)` for `increase`. Run a one-sided Wilcoxon in the expected direction, and the symmetric two-sided as secondary.

2. **No dedicated bidirectionality figure.** `fig1` shows all 11 condition curves overlaid (crowded); `fig4` is a single bar chart ranked by best logit (no split between decrease-expected and increase-expected; no grouping). A reviewer wants a symmetric bar chart: critical roles reducing, conformist roles increasing, axis in the middle, random band at zero. Nothing in the repo draws this.
   - **Fix:** new `paper/scripts/fig_bidirectionality.py` that reads `sycophancy_rates_test.json` + `multiseed_aggregate.json` and produces a two-sided bar chart, critical on the left, conformist on the right, with multi-seed error bars.

3. **Dose-response Spearman is one-tailed in implementation but reported two-sided.** `tests["dose_response_row_level"]` (L614–636) runs `scipy.stats.spearmanr` which is two-sided. For conformist roles the expected dose-response is **positive** (more coefficient → more sycophancy). Needs a signed/one-sided report, or at minimum a sign-consistency check between critical and conformist.

4. **Snag S4 re-stated (from PIPELINE_MAP snag list):** `over_correction_check.py` hardcodes `+2000` for every condition. For conformist roles (Objective 1) we want to test the increase at the condition's best *positive* coef, which may not be +2000. For critical roles at best coef > or < 2000, the structured probe is at the wrong operating point.

5. **Multi-seed error bars not on fig1 / fig4.** `run_multiseed.py` produces `results/multiseed_aggregate.json` with cross-seed mean/std per condition, but `03_analysis.py` does not read it. `fig1` and `fig4` show a single seed; the "error bars" on `fig4` are within-seed bootstrap CIs, not across-seed. This is a direct miss on the paper requirement.

### Severity

High. (1) and (2) are blockers for a clean Objective-1 claim; (5) blocks promoting anything to a headline figure.

---

## Objective 2 — CAA (Rimsky et al. 2024) as targeted baseline, head-to-head vs axis vs best role

### What the code does

- `02b_extract_caa.py` extracts CAA from `sycophancy_on_nlp_survey` + `sycophancy_on_political_typology_quiz` (disjoint from philpapers eval). Saves `caa_unit.pt`, metadata, cosine with assistant axis.
- `caa` is treated as one of the 11 real conditions throughout `02_evaluate_steering.py` / `03_analysis.py` — same Wilcoxon, same dose-response, same appearance in `fig1`/`fig4`/`fig5`/`fig6`.

### What is missing or broken

1. **No head-to-head figure. `fig4` is a lump, not a comparison.** `fig4` (`03_analysis.py` L378–452) is a bar chart over **all 11 real conditions** with bootstrap 95 % CIs, ordered by the arbitrary `CONDITIONS_REAL` list, colour-coded by `COLORS`. There is no isolated panel comparing CAA vs assistant_axis vs best role. No matched-effect-size comparison. The sign-convention caveat (CAA points honest→syc so uses *negative* coefficient) is noted in the README and `02b_extract_caa.py` but **not visualised** — CAA and axis/role bars sit side by side with different-sign "best coefficients" and no annotation of the sign convention on `fig4`. A reviewer scanning the figure will misread CAA.
   - **Fix:** new `paper/scripts/fig_caa_headtohead.py` that reads `sycophancy_rates_test.json` + `multiseed_aggregate.json` and emits a 3-bar (or 4-bar) figure: baseline, assistant_axis@best, CAA@best, best_role@best, with ±std across seeds, with a sign-convention note in the caption.

2. **Matched-operating-point comparison not computed.** The paper wants "at X % reduction, which vector uses lower |coefficient|?" — i.e. efficiency. `03_analysis.py` L693–708 (`tests["efficiency"]`) computes `reduction / |best_coefficient|`, which is a *single* point per vector, not a matched operating-point comparison. No iso-effect interpolation exists.
   - **Fix:** small plotting/analysis script that interpolates each condition's coefficient→logit curve from `sycophancy_rates_*.json` and, at each target logit, reports the minimum |coef| across vectors. Produce a line plot of |coef| needed vs target sycophancy reduction.

3. **Train/eval leakage on CAA is not **tested**, only asserted.** The README (K2 design decisions) and `02b_extract_caa.py` claim the CAA training sets (`nlp_survey`, `political_typology_quiz`) are disjoint from the philpapers eval. Nothing in the code checks for overlap by question text, orig_id, or hash. Low probability of real leakage given different dataset families, but a reviewer will ask — include a check in `paper/scripts/verify_caa_leakage.py` that hashes CAA training prompts and eval prompts, reports zero overlap, and save the receipt to `paper/tables/`.

4. **Snag S1 (config filename mismatch) halves the CAA training set.** `config.py` L57 lists `sycophancy_on_political_typology.jsonl`; the actual file (per `anthropics/evals` and HF) is `sycophancy_on_political_typology_quiz.jsonl`. `02b_extract_caa.py` L49–52 prints "WARNING: ... not found, skipping" and continues with only the NLP-survey half. This is load-bearing: the CAA vector as currently defined is trained on ~half the intended data, so the CAA-vs-role comparison at the heart of Objective 2 is not the intended comparison. **Must be fixed before extracting CAA.**
   - **Fix:** edit `config.CAA_DATASETS` to the correct filename. Smallest possible change, already isolated to one constant.

5. **No CAA-extraction error bars.** Orthogonal to what `run_multiseed.py` does (eval noise). CAA is extracted from a single pass over the full NLP-survey + political-typology pairs — no resampling of CAA training pairs, so we cannot report how stable the `caa` direction itself is. Not strictly required for the paper but a reviewer may raise it.
   - **Out of scope for Phase 2 unless time permits.** Worth flagging in LIMITATIONS.md.

### Severity

High. (1) and (4) block the Objective-2 claim. (4) must be fixed before any GPU run or all downstream `caa` numbers are on the wrong vector.

---

## Objective 3 — Compare direction vectors: cosine geometry, head-to-head, CAA-aligned / residual decomposition

### What the code does

- **Geometry:** `01_prepare_steering_vectors.py` L118–149 computes the full pairwise cosine matrix, saves `results/vector_cosine_similarities.json`. `fig5` (cosine heatmap) is drawn in `03_analysis.py` L455–479. `fig6` (PCA persona space) at L713–781 plots every unit vector with its best-logit annotated.
- **Decomposition (geometric):** `01_prepare_steering_vectors.py` L93–116 projects each role onto CAA, saves `caa_decomposition.json` with `cosine_with_caa`, `caa_component_norm`, `residual_norm`. It also **saves unit-normed CAA-component and residual vectors** to `vectors/steering/{role}_caa_component_unit.pt` and `{role}_residual_unit.pt`.
- **Decomposition (behavioural), attempted:** `03_analysis.py` L678–688 tries to look up rows for `(f"{cond}_caa_component", best_c)` and `(f"{cond}_residual", best_c)` in `idx` and report their logit/rate.

### What is missing or broken (the big one)

1. **The behavioural decomposition never runs.** `config.py` defines `CONDITIONS = CONDITIONS_REAL + CONDITIONS_RANDOM` — i.e. the 21 conditions the paper expects. It does **not** include any `{role}_caa_component` or `{role}_residual` condition. `02_evaluate_steering.py` therefore never steers with those vectors, so `idx[(f"{cond}_caa_component", best_c)]` is always empty in `03_analysis.py` L684. The `caa_decomposition` tests block in `03_analysis.py` ends up with only the geometric cosine and norms — no per-role behavioural split. **This is the core missing result for Objective 3.** The repo has the vectors saved to disk but never evaluates them.
   - **Fix:** the cleanest option is to add a new evaluation path. Either (a) extend `CONDITIONS` to include one `{role}_caa_component` and one `{role}_residual` for every role (11 extra × 12 coeffs = ~$25 and ~8 extra GPU hours), or (b) write a focused `02c_evaluate_decomposition.py` that only evaluates the decomposition vectors at each role's best coefficient from the tune split (11 × 2 decomp vectors × 1 coef × 600 rows ≈ 40 min on H100, ~$2). Option (b) is cheaper and enough for the paper; option (a) gives full dose-response curves for the decomposition and is nicer if budget allows. Recommended default: (b), with the best coef transferred from the tune-split `03_analysis.py` output.

2. **No decomposition figure.** Even once (1) is fixed, there is no plotting code for the headline result. A reviewer needs one figure that makes the following claim visible: *for each role, here is the fraction of the behavioural sycophancy reduction that survives projection onto CAA, and here is the fraction carried by the residual.* Without this figure, Objective 3 is not present in the paper.
   - **Fix:** new `paper/scripts/fig_decomposition.py` producing a stacked/dodged bar chart per role: three bars per role (full, CAA-component, residual) showing Δlogit from baseline, ordered by cos(role, CAA). Should also explicitly flag cases where `|residual_reduction| ≈ 0` (role behaviour is dominated entirely by its CAA-aligned component) and where `|residual_reduction| ≈ |full_reduction|` (role behaviour is orthogonal to CAA and works through something else). These are the honest-negative-result cells and must appear.

3. **`caa_decomposition.json` saves `caa_component` as a *unit-normed* vector but the norm fraction is what carries the headline.** `01_prepare_steering_vectors.py` L105 does `unit(caa_component)` before saving. The geometric decomposition is faithful because the JSON also records `caa_component_norm` and `residual_norm` — but the saved `.pt` files are both unit-norm, so when (1) is fixed and they are steered at coefficient `c`, the effective steering magnitude along CAA and the residual are decoupled from their original norms. This is actually what we want for the behavioural test (unit inputs, sweep coefficient), but the paper text must be precise: "we steer with *the unit-normalised CAA-aligned component and residual of each role*, at matched coefficient," not "we steer with the components themselves."

4. **Geometry figures exist but are not paper-ready:**
   - `fig5` (cosine heatmap) includes `random_0` but not the decomposition vectors and not the conformist roles in a separate block. The re-ordering for publication (group: axis, critical, conformist, caa; then randoms) is not done.
   - `fig6` (PCA) at L713–781 runs but the annotation format (`label + "\n(logit +X.XX)"`) has no error bar on the logit and conflates vectors with very different steering efficiencies. For paper, annotate with Δlogit from baseline ± seed std, and mark the condition symbol (critical/conformist/caa/random) legibly.

5. **Response-token vs prompt-token projection is captured but not compared.** `02_evaluate_steering.py` L85–95, 144–172 captures `response_projection` for every row and saves it to `axis_projections_*.json`. `03_analysis.py` only uses `axis_projection` (prompt-last token). The "is response-token projection a better predictor?" claim from the changelog is entirely untested in the stats output. Low-severity for Objectives 1–3 as stated (more relevant to a mechanistic sub-section), but the user flagged it explicitly.
   - **Fix:** add a short block to `03_analysis.py` (or a separate `paper/scripts/projection_ablation.py`) computing pooled Pearson r for both projections and reporting which is larger with 95 % CI on the difference.

### Severity

Critical for Objective 3. (1) and (2) together mean the paper has no evidence for the decomposition claim at all today.

---

## Process-level gaps (apply to all three objectives)

These are the cross-cutting issues that break reproducibility or reviewer confidence regardless of which objective.

### P1. Tune/test hygiene is not enforced in all reported numbers

- `03_analysis.py`'s `best_per_cond` (L382–399) reads `rates[cond]` for the split it was invoked on. If run as `--split tune`, best coef is computed on tune — correct. If run as `--split test`, best coef is computed on **test** — **this is selection on the held-out set**, contaminating the reporting. The tune/test discipline is only enforced if the workflow is: run `03 --split tune`, hand-transcribe best coefs, then run `03 --split test` with those coefs locked in. Nothing in the code enforces this.
   - **Fix:** `03_analysis.py --split test` should read the tune-split's `best_per_cond` from `results/summary_tune.txt` or a new `results/best_coefs_tune.json`, and only evaluate at those locked coefs. Small targeted edit in `03_analysis.py`. Also: bundle best-coef selection into a separate helper that writes `results/best_coefs_{split}.json`, so downstream analysis reads from there.
   - Every numerical claim in the paper must cite the `_test` file. Flag any tune-only numbers explicitly.

### P2. `over_correction_check.py` and `qual_check.py` read legacy no-suffix files

- Snag S3: `over_correction_check.py` L40 reads `results/all_results.json`; `qual_check.py` L26 reads `results/checkpoints/baseline.json`. After a tune/test run these files don't exist. The overcorrection analysis either needs a `--split` flag passed through, or the simplest fix is to always also write no-suffix aliases.
   - **Fix:** add `--split` to both scripts, defaulting to `test` (since overcorrection / qualitative are reportable). Condition selection in `over_correction_check.py` should also read best coef per condition from `results/best_coefs_test.json` instead of hardcoded `+2000` — see next.

### P3. `over_correction_check.py` hardcodes `+2000` as the operating point

- Snag S4: `02_evaluate_steering.py` does a full sweep, `03_analysis.py` picks a best coef per condition, and `over_correction_check.py` ignores both and uses +2000 for every condition. For conditions whose best coef is ±500 or ±5000, the overcorrection numbers are measured at the wrong regime and cannot be compared fairly.
   - **Fix:** read `results/best_coefs_test.json` (once P1 writes it) and pass each condition's actual best coef.

### P4. `03_analysis.py` writes figure filenames without a split suffix

- Snag in updated pipeline map: `fig1..fig6.png` are overwritten when the script is run first on tune then test. This means you cannot retain both sets.
   - **Fix:** either add the suffix to figure filenames in `03_analysis.py`, or write paper figures into `paper/figures/` from the paper scripts and leave `figures/` as scratch. The latter is cleaner and matches the deliverable spec.

### P5. `00_setup.py` does not download the role vectors or sycophancy datasets

- Snag S2: setup reads raw tensors from `vectors/gemma-2-27b/` and `data/sycophancy/` but never downloads them. The `.gitignore` implies `vectors/gemma-2-27b/` comes from `lu-christina/assistant-axis-vectors` on HF. The sycophancy datasets come from `anthropics/evals` on GitHub.
   - **Fix:** add a one-shot download helper (prefix `00_setup.py` with an HF + GH download block, or ship a separate `scripts/fetch_external.py`) that pulls all required files and checks sha sums. README reproducibility section must point at this.

### P6. Multi-seed results do not flow into the headline figures or the summary

- `run_multiseed.py` produces `results/multiseed_aggregate.json` with per-condition mean/std. `03_analysis.py` never reads it. `summary.txt` never reports multi-seed numbers. Paper figures must show cross-seed bars; per-seed bootstrap (`fig4`) is not a substitute.
   - **Fix:** paper figure scripts read `multiseed_aggregate.json` for error bars when present, fall back to single-seed bootstrap with a caveat.

### P7. Over-correction taxonomy is computed but not plotted

- Snag in updated map: `over_correction_check.py` produces the full AGREE/DISAGREE × CORRECT/WRONG + HEDGE + REFUSE taxonomy but prints a wide text table and saves JSON. There is no per-condition stacked bar chart. Without the figure, "reducing sycophancy" is indistinguishable from "becoming incoherently contrarian" in the paper.
   - **Fix:** new `paper/scripts/fig_overcorrection_taxonomy.py`: per-condition stacked bar, categories in fixed legible order, true-claim and false-claim panels side by side.

### P8. MCC family is too narrow

- `03_analysis.py` applies Holm-Bonferroni *within* each test family (`primary_wilcoxon_best_vs_baseline`: 11 tests; `mcnemar_best_vs_baseline`: 11 tests). It does **not** correct across the full set of claims the paper makes (21 conditions × {primary Wilcoxon, dose-response Spearman, real-vs-random} ≈ 50 tests). Not a bug — a reviewer will question whether per-family Holm is enough.
   - **Decision:** keep per-family Holm as primary (standard in the subfield) but add a table row reporting the global Holm or BH-FDR across the full set, to pre-empt the reviewer.

### P9. Degradation flags exist but aren't shown

- `results/degradation_flags.json` is written but not surfaced in `summary.txt` or any figure. If a condition's "best" cell is degraded, the paper must not report it as a reduction.
   - **Fix:** in `03_analysis.py` summary, skip conditions flagged degraded at their best coef and explicitly note them.

---

## Prioritised fix list for Phase 2

Ordered by what will most damage the paper if missed. File references are for Phase 2 work.

1. **Fix `config.CAA_DATASETS` filename** (`config.py` L55–58). Blocks any CAA-based result.
2. **Add behavioural decomposition evaluation** (new `scripts/02c_evaluate_decomposition.py` or extend `CONDITIONS`). Objective 3 has no result without this.
3. **Fix best-coef selection for conformist roles** (`03_analysis.py` L382–399) with a direction-aware selector, and add one-sided Wilcoxon. Objective 1 bidirectionality has no result without this.
4. **Lock best coef on tune and reuse on test** (new `results/best_coefs_tune.json` + `03_analysis.py --split test` reads it). Selection bias currently contaminates test numbers.
5. **Headline figures not produced today:**
   - `paper/scripts/fig_bidirectionality.py` (Obj 1).
   - `paper/scripts/fig_caa_headtohead.py` (Obj 2).
   - `paper/scripts/fig_decomposition.py` (Obj 3).
   - `paper/scripts/fig_overcorrection_taxonomy.py` (overcorrection).
6. **Multi-seed plumbing into paper figures** — paper figure scripts read `multiseed_aggregate.json` when present.
7. **Over-correction best-coef-per-condition + split-aware IO** (`over_correction_check.py`).
8. **Response-token vs prompt-token projection comparison** (small analysis block).
9. **`00_setup.py` download helper + README reference list** (reproducibility).
10. **Split-suffix on figures or move paper figures under `paper/`** (`03_analysis.py`, cosmetic but necessary for storing tune and test artefacts side by side).
11. **CAA train/eval leakage receipt** — cheap, helps reviewers.
12. **Degradation flags folded into summary** — cheap.

Items 1 and 3 are corrections to existing scripts and are the only places in this plan that modify shipped pipeline code. Everything else is additive (new paper-scoped scripts) so the main pipeline stays frozen and re-runs cheaply.

## Open questions for you before Phase 2

- **Behavioural decomposition eval cost:** is the cheap option (b) — "eval each role's two decomposition vectors only at the role's tune-split best coef, ~40 min total" — acceptable, or do you want (a), "full dose-response sweep of all decomposition vectors, ~8 h"? Option (b) is enough for a headline bar chart; (a) buys you a decomposition dose-response figure on top.
- **Multi-seed budget:** 5 seeds × (tune+test) is ~30–40 h. 5 seeds × tune only is ~15–20 h. 3 seeds × test only is ~10–12 h. Which?
- **Overcorrection re-run:** the probe set in `over_correction_check.py` is 80 generations. Re-running at condition-specific best coef (≥5 conds if we keep the current selection; potentially 11 real conds if we expand) is ≤1 GPU hour. Assume expand to all 11 real conds?

Once these are settled I move to Phase 2 and start executing fixes.
