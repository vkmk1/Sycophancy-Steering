# Sycophancy steering — Qwen 3 32B replication

Full multi-seed replication of the Gemma-2-27b-it sycophancy-steering
pipeline (https://github.com/kelkarI/sycophancy-final) on **Qwen/Qwen3-32B**
at layer 32. Includes three standalone CAA-orthogonal residual conditions
matching the Gemma addition.

## Condition set (24 total, Holm applied across all)

- `assistant_axis` — targeted axis vector from `lu-christina/assistant-axis-vectors`
- 5 critical roles: `devils_advocate, contrarian, skeptic, judge, scientist`
- 4 conformist roles: `peacekeeper, pacifist, collaborator, facilitator`
- `caa` — Rimsky et al. 2024 sycophancy direction, extracted locally on Qwen
- 10 `random_i` controls
- 3 standalone residuals (this experiment's addition):
  `skeptic_residual, contrarian_residual, devils_advocate_residual`

The third residual (devils_advocate) is the role with the
largest \|cos(role, CAA)\| on Qwen (0.108), matching the user's per-model
"pick the high-cos role" rule (on Gemma the same rule picked collaborator
at 0.165).

## Model-specific adaptations vs. Gemma

| | Gemma | Qwen |
|---|---|---|
| target layer | 22 / 46 | 32 / 64 (canonical from `assistant_axis.models.MODEL_CONFIGS`) |
| coefficient sweep | ±500, ±1000, ±2000, ±5000 | ±50, ±100, ±200, ±500 (10× smaller; Qwen activation norms require it) |
| chat-template flag | — | `enable_thinking=False` — Qwen 3's template injects a `<think>` block otherwise |
| hidden size | 4608 | 5120 |

## Seeds

Matching the Gemma pipeline: 5 tune seeds (42, 7, 123, 456, 789) → 3 test
seeds (42, 7, 123). Best coefficients locked from the mode-across-tune-seeds
aggregate in `results/best_coefs_tune_aggregate.json`.

## Headline numbers (test split, Holm across 24, 3 seeds)

Baseline sycophancy logit across 3 test seeds: +3.00 ± 0.14. Baseline
sycophancy rate: 84 ± 1.5 %.

| Condition | mean Δ logit (3 seeds) | sig seeds |
|---|---|---|
| CAA (targeted) | −1.97 | 3/3 |
| assistant_axis | −2.41 | 3/3 |
| devils_advocate | −2.27 | 3/3 |
| contrarian | −2.28 | 3/3 |
| skeptic | −1.82 | 3/3 |
| judge | −1.70 | 3/3 |
| scientist | −0.98 | 3/3 |
| **devils_advocate ⊥ CAA** | −2.19 | 3/3 |
| **contrarian ⊥ CAA** | −1.65 | 3/3 |
| **skeptic ⊥ CAA** | −1.80 | 3/3 |
| peacekeeper (conformist) | −0.71 | 0/3 |
| collaborator (conformist) | −0.03 | 0/3 |
| facilitator (conformist) | −0.47 | 0/3 |
| pacifist (conformist) | −2.98 | 0/3 (degraded, rate 50% at +500) |

## Full reproduction

```bash
# From this repo root, with Qwen 3 32B access on HF and the
# safety-research/assistant-axis library at $ASSISTANT_AXIS_PATH:

export EXPERIMENT_ROOT=$PWD
export ASSISTANT_AXIS_PATH=/path/to/assistant-axis
cd scripts

# Eval data (sycophancy_on_philpapers2020):
python 00b_rebuild_eval.py

# Steering vectors:
python extract_all_vectors.py --skip-personas       # CAA extraction on Qwen
python build_vectors_from_official.py               # load qwen-3-32b/ from HF dataset + decompose

# Full multi-seed tune (~6 h on GH200):
python run_multiseed.py --seeds 42 7 123 456 789 \
                        --split tune --skip-response-gen

# Aggregate best coefficient across tune seeds:
python aggregate_multiseed.py --split tune

# Multi-seed test with locked coefs (~3.5 h):
python run_multiseed.py --seeds 42 7 123 --split test \
    --locked-coefs-from ../results/best_coefs_tune_aggregate.json \
    --skip-response-gen

# Aggregate across test seeds:
python aggregate_multiseed.py --split test

# Auxiliaries:
python 02c_evaluate_decomposition.py --split test \
    --locked-coefs-from ../results/best_coefs_tune_aggregate.json
python over_correction_check.py \
    --locked-coefs-from ../results/best_coefs_tune_aggregate.json
python residual_standalone_report.py --split test
```

## Vector provenance

- `assistant_axis`, 9 role vectors: loaded from
  `lu-christina/assistant-axis-vectors` HF dataset, `qwen-3-32b/` subdir, at
  layer 32. Each role direction is `role_vector[L] − default_vector[L]`,
  unit-normalised.
- `caa`: extracted locally on Qwen 3 32B (Rimsky et al. 2024 contrast on
  2000 sycophancy training pairs at layer 32).
- 10 random controls: unit-Gaussian, seeds `RANDOM_SEED_BASE + i`.
- Residuals: `role_vector − proj_CAA(role_vector)`, unit-normalised.

Raw vectors are gitignored (see `.gitignore`); re-generate with the two
extraction scripts. Metadata lives in
`vectors/steering/persona_extraction_metadata.json` and `caa_metadata.json`.
