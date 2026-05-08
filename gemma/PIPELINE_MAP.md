# PIPELINE_MAP.md

One-page map of the sycophancy-steering pipeline. Nothing here is executed yet; this is the read-only survey from Phase 0.

## Model, eval, steering

- **Model:** `google/gemma-2-27b-it`, bf16, `device_map="auto"`. `TARGET_LAYER = 22`; hook on `model.model.layers[22]`.
- **Answer tokens:** `A = 235280`, `B = 235305` (single-token; verified in `00_setup.py`).
- **Steering:** `assistant_axis.steering.ActivationSteering` from `safety-research/assistant-axis` (addition at every position).
- **Eval set:** `sycophancy_on_philpapers2020` → 300 two-choice base questions, each run in A/B and swapped orderings → 600 rows. `base_id` pairs the two variants.
- **CAA training set:** `sycophancy_on_nlp_survey` + `sycophancy_on_political_typology_quiz` (disjoint from eval).
- **Conditions (21):** `assistant_axis`, 5 critical roles (`devils_advocate, contrarian, skeptic, judge, scientist`), 4 conformist roles (`servant, diplomat, disciple, peacemaker`), `caa`, and `random_0..random_9`.
- **Coefficient sweep:** `[-5000, -3000, -2000, -1000, -500, -200, 0, 200, 500, 1000, 2000, 3000, 5000]`.
- **Tune/test split:** 50/50 at base-question level, `TUNE_TEST_SEED = 99`; pairs stay together.

## Canonical execution order

Runtime estimates assume a single H100 80GB (SXM) with the model already cached in `~/.cache/huggingface/` after the first load. First-ever model download adds ~10 min (~55 GB of weights).

| # | Script | GPU? | H100 runtime | Reads | Writes | Notes |
|---|---|---|---|---|---|---|
| 1 | `00_setup.py` | yes | ~10 min first load; ~2 min cached | `vectors/gemma-2-27b/{assistant_axis,default_vector}.pt`, `vectors/gemma-2-27b/role_vectors/*.pt`, `data/sycophancy/sycophancy_on_philpapers2020.jsonl` | `data/eval_data_raw.json`, `data/setup_info.json`, `data/target_layer.txt` | Loads model; 5-row sanity check. Does **not** download role vectors or syco datasets — they must exist first. |
| 2 | manual curl / HF download | no | ~2 min | — | `data/sycophancy/*.jsonl`, `vectors/gemma-2-27b/*.pt` | README only documents the philpapers curl. See "External dependencies" below. |
| 3 | `00b_rebuild_eval.py` | no | <10 s | `data/sycophancy/sycophancy_on_philpapers2020.jsonl` | `data/eval_data.json` (600 rows, counterbalanced) | `--seed` / `--n-base` flags for multi-seed subsampling. |
| 4 | `02b_extract_caa.py` | yes | ~25–35 min (two forward passes × ~1–2 k pairs) | `data/sycophancy/{nlp_survey,political_typology_quiz}.jsonl` | `vectors/steering/caa_unit.pt`, `vectors/steering/caa_metadata.json` | Rimsky et al. CAA; last-token activation, mean(syc) − mean(honest). |
| 5 | `01_prepare_steering_vectors.py` (re-run after 02b) | no | <5 s | raw vectors + `caa_unit.pt` | `vectors/steering/*_unit.pt`, `*_caa_component_unit.pt`, `*_residual_unit.pt`, `vectors/steering/caa_decomposition.json`, `results/vector_cosine_similarities.json` | Second run is what puts CAA into the cosine matrix and writes the decomposition. |
| 6 | `02_evaluate_steering.py --split tune` | yes | ~3–4 h (21 conds × 12 non-zero coeffs × 300 rows × 2 passes/row) | `data/eval_data.json`, `vectors/steering/*_unit.pt` | `results/all_results_tune.json`, `sycophancy_rates_tune.json`, `degradation_flags_tune.json`, `axis_projections_tune.json`, `tune_test_split.json`, `results/checkpoints/*_tune.json` | Per-row: `logprob_A/B`, `syc_logit`, `axis_projection`, `response_projection` (mean over 8 greedy gen tokens). `--resume` honours checkpoints. |
| 7 | `02_evaluate_steering.py --split test` | yes | ~3–4 h | same | `*_test.json` counterparts | Held-out final numbers. |
| 8 | `03_analysis.py --split tune` | no | ~1–3 min CPU | all `*_tune.json` + cosine + decomposition | `figures/fig{1..6}.png`, `results/summary_tune.txt`, `results/statistical_tests_tune.json` | **Figure filenames carry no split suffix**, so running 03 on test overwrites the tune figures. |
| 9 | `03_analysis.py --split test` | no | ~1–3 min CPU | `*_test.json` | same filenames with `_test` suffix | Reportable numbers. |
| 10 | `over_correction_check.py` | yes | ~25–40 min (16 probes × 5 conds × ~150 gen tokens) | `results/all_results.json` (legacy, no suffix) + `vectors/steering/*_unit.pt` | `results/over_correction_eval.json` | 16 probes × 5 conditions at `+2000`; keyword classifier over generated text. |
| 11 | `run_multiseed.py --seeds 42 7 123 456 789` | yes | ~(2 + 6–8 h) × N_seeds ≈ 30–40 h for 5 seeds on full split | runs 00b/02/03 per seed | `results/seed_<k>/*`, `results/multiseed_aggregate.json` | Thin orchestrator; re-subsamples base questions and re-runs. Does **not** re-run 02b or 01 — see below. |

### Rough first-H100 budget

| Scope | GPU hours | @ $2.49/hr |
|---|---|---|
| Single-seed, tune + test (02 × 2) + 02b + qualitative | ~8–10 h | ~$20–25 |
| + 5-seed multiseed on full split (same 02b reused) | +30–40 h | +$75–100 |
| + 5-seed multiseed on tune split only | +15–20 h | +$40–50 |

### CAA extraction sharing across seeds (confirmed)

`run_multiseed.py` runs only `00b_rebuild_eval.py → 02_evaluate_steering.py → 03_analysis.py` per seed (`scripts/run_multiseed.py` lines 66–83). It does **not** call `02b_extract_caa.py` or `01_prepare_steering_vectors.py` inside the per-seed loop. So **the CAA vector is extracted once up-front and reused across all seeds**, and multi-seed error bars for the `caa` condition reflect only eval-sampling (and random-vector) noise — not CAA-extraction noise. This is the correct behaviour for the stated research question: we want to know how stable the `caa` result is across eval subsamples, holding the targeted vector fixed. Extraction-noise error bars would require a separate re-extraction loop, which no script currently implements.

Implication — ordering prereq: `run_multiseed.py` assumes `vectors/steering/caa_unit.pt`, the `*_caa_component_unit.pt`, and `*_residual_unit.pt` files already exist. If they don't, `02_evaluate_steering.py` errors on the `caa` condition. This prereq is not enforced or documented.

Auxiliary, not on the critical path: `pilot_debug.py` (hook-sanity), `pilot_scale.py` (coefficient regime finder), `qual_check.py` (free-form comparison on 5 high-sycophancy prompts), `find_skeptic_agreement.py` (diagnostic).

## Figure provenance

All six figures are produced by `03_analysis.py`:

| Figure | Source code location (03_analysis.py) | Reads |
|---|---|---|
| fig1 steering curves | ~L136–199 | `sycophancy_rates{suffix}.json` |
| fig2 projection dists | ~L203–291 | baseline rows from `all_results{suffix}.json` |
| fig3 projection shift | ~L340–375 | `all_results{suffix}.json` + `vector_cosine_similarities.json` |
| fig4 best-per-condition | ~L378–452 | `sycophancy_rates` + bootstrap over `all_results` |
| fig5 cosine heatmap | ~L455–479 | `vector_cosine_similarities.json` |
| fig6 persona-space PCA | ~L713–781 | all `vectors/steering/*_unit.pt` |

CAA decomposition is computed in `01_prepare_steering_vectors.py` and *summarized* in `03_analysis.py` L664–708 (`tests["caa_decomposition"]`, `tests["efficiency"]`). It is written to `statistical_tests{suffix}.json` but is **not drawn as a figure** — that's a Phase 1 gap.

Over-correction taxonomy is computed in `over_correction_check.py` and saved to `results/over_correction_eval.json` as `summary_by_condition` (per-condition counts of AGREE_CORRECT / DISAGREE_CORRECT / AGREE_WRONG / DISAGREE_WRONG / HEDGE / REFUSE). It is **not plotted** — that's a Phase 1 gap.

## External dependencies

Required, not all linked from README:

1. **Model weights** — `google/gemma-2-27b-it` on HuggingFace (gated; accept the Gemma license). Linked in README.
2. **Steering library** — `safety-research/assistant-axis` on GitHub (provides `assistant_axis.steering.ActivationSteering`). Linked in README.
3. **Pre-computed role + axis vectors** — `lu-christina/assistant-axis-vectors` on HuggingFace, per `.gitignore`. Used as `vectors/gemma-2-27b/{assistant_axis,default_vector}.pt` and `vectors/gemma-2-27b/role_vectors/{role}.pt` for the 5 critical + 4 conformist roles. **Not linked from README; `00_setup.py` does not download them.**
4. **Sycophancy eval dataset (philpapers2020)** — `anthropics/evals` GitHub raw blob `5525210614d4f26b1732042e7dcb7210d23fe5aa`. Linked in README.
5. **CAA training datasets** — `sycophancy_on_nlp_survey.jsonl` and `sycophancy_on_political_typology_quiz.jsonl` from `anthropics/evals`. Also mirrored on `Anthropic/model-written-evals` HF dataset (but the HF mirror duplicates nlp_survey under the philpapers name — don't use HF for philpapers). **Not linked from README.**
6. **CAA reference** — Rimsky et al. 2024, "Steering Llama 2 via Contrastive Activation Addition" (arXiv:2312.06681). Cited in docstrings but not listed in README.

## Known pipeline snags to flag in Phase 1

- `config.py` refers to `sycophancy_on_political_typology.jsonl`; the real filename is `sycophancy_on_political_typology_quiz.jsonl`. `02b_extract_caa.py` will silently skip that file (it just warns and continues), halving the CAA training set without failing loudly.
- `00_setup.py` reads raw role vectors from `vectors/gemma-2-27b/` but never downloads them — the user has to fetch them from `lu-christina/assistant-axis-vectors` manually. The `.gitignore` says so; the README does not.
- `over_correction_check.py` reads `results/all_results.json` (legacy no-suffix). If only `--split test/tune` was run, the no-suffix file is absent and Part 1's distribution check silently skips.
- `qual_check.py` reads `results/checkpoints/baseline.json` (legacy no-suffix). Same issue.
- `over_correction_check.py` hardcodes `+2000` as the operating point for every condition. Best coefficient is actually condition-dependent per `03_analysis.py`'s `best_per_cond` — this is a coefficient-matching mismatch at comparison time.
- Response-token projections are captured and stored, but `03_analysis.py` does not currently compare `response_projection` vs `axis_projection` as predictors.
- Conformist roles exist and are evaluated, but `fig4` only includes whatever wins "excess_over_random_mean" — there is no dedicated bidirectionality figure.
