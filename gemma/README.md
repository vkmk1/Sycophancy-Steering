# General Persona Directions Reduce Sycophancy via Activation Steering

## Abstract

We study whether **general persona directions** extracted from open-ended
role-playing (skeptic, devil's advocate, scientist, judge, contrarian, plus
four conformist roles) can steer a model away from sycophancy as effectively
as a **targeted Contrastive Activation Addition** (CAA, Rimsky et al. 2024)
vector trained on sycophancy-specific data, in Gemma 2 27B Instruct. We
report three findings on the philpapers2020 sycophancy benchmark with a
tune/test split, multi-seed robustness, and Holm-Bonferroni correction:

1. **Role directions move sycophancy bidirectionally.** Steering toward
   critical roles reduces sycophancy; steering toward conformist roles
   (`servant`, `diplomat`, `disciple`, `peacemaker`) produces the symmetric
   *increase*. The effect is not a noise artifact of adding any unit vector
   to layer 22: 10 random controls define a null band whose width does not
   overlap with the real-condition effect at tune-locked best coefficients.
2. **A general persona direction matches a targeted CAA vector.** At
   matched operating points on the held-out test split, the best critical
   role direction is within a small margin of CAA on Δlogit and close in
   efficiency per unit steering magnitude. Training data for the two is
   disjoint (verified by hash-overlap receipt).
3. **The *direction* carries the effect, not the magnitude that happens
   to align with CAA.** Decomposing each role into its CAA-aligned
   component and its residual, and steering with each at the role's best
   coefficient with both pieces **unit-normalised**, reveals per-role
   which of the two carries the behavioural reduction. Some roles are
   dominated by their CAA-aligned component; others retain most of the
   effect in their residual, indicating an independent mechanism.

See `paper/RESULTS.md` for the full walk-through with file-traced numbers,
`paper/METHODS.md` for model and statistical detail, and
`paper/LIMITATIONS.md` for caveats.

---

## References and external dependencies

| Resource | Purpose | Link |
|---|---|---|
| `google/gemma-2-27b-it` | Model under study (gated, accept the Gemma license on HF). | https://huggingface.co/google/gemma-2-27b-it |
| `safety-research/assistant-axis` | `ActivationSteering` context manager used to apply steering hooks. | https://github.com/safety-research/assistant-axis |
| `lu-christina/assistant-axis-vectors` | Pre-computed `assistant_axis.pt`, `default_vector.pt`, and per-role residual means for Gemma 2 27B. Downloaded by `scripts/fetch_external.py`. | https://huggingface.co/lu-christina/assistant-axis-vectors |
| `anthropics/evals` (GitHub) | Sycophancy A/B evaluations: `sycophancy_on_philpapers2020.jsonl` (our held-out eval), `sycophancy_on_nlp_survey.jsonl` and `sycophancy_on_political_typology_quiz.jsonl` (our CAA training set). We pin immutable blob SHAs. | https://github.com/anthropics/evals/tree/main/sycophancy |
| Rimsky et al. 2024 | CAA method. | https://arxiv.org/abs/2312.06681 |
| Perez et al. 2023 | Model-Written Evaluations (source of the philpapers2020 A/B preferences). | https://arxiv.org/abs/2212.09251 |

---

## Reproducibility — exact commands on a single H100

All commands run from `/home/ubuntu/experiment/scripts/`. Approximate
runtimes are for an H100 80GB with the model already cached. See
`PIPELINE_MAP.md` for the full per-step runtime breakdown and
`paper/scripts/build_all.py` for a single-command rebuild of every paper
artefact.

```bash
# ---- setup ----
python fetch_external.py                       # <5 min; downloads vectors + datasets
python 00_setup.py                             # ~10 min; loads model, sanity checks
python 00b_rebuild_eval.py                     # <10 s; counterbalanced eval_data.json

# ---- vectors ----
python 02b_extract_caa.py                      # ~30 min; CAA vector
python 01_prepare_steering_vectors.py          # <5 s;  decomposition + cosine matrix

# ---- tune split (multi-seed) ----
python run_multiseed.py --seeds 42 7 123 456 789 --split tune
# ~20 h on H100 for all 5 seeds; saves per-seed results under results/seed_*/

# ---- aggregate tune best coefs ----
python ../paper/scripts/aggregate_multiseed.py --split tune

# ---- test split (multi-seed, locked coefs) ----
# Note: run_multiseed.py re-runs 02_evaluate_steering.py per seed; the
# tune-split locked coefs are read by 03_analysis.py, which run_multiseed
# invokes. For the multi-seed test runs, we pass the tune aggregate as
# the lock source in 03_analysis --locked-coefs-from.
python run_multiseed.py --seeds 42 7 123 --split test
python ../paper/scripts/aggregate_multiseed.py --split test

# ---- decomposition (single run, tune-locked) ----
python 02c_evaluate_decomposition.py \
    --split test \
    --locked-coefs-from ../results/best_coefs_tune.json
# ~40 min on H100

# ---- overcorrection ----
python over_correction_check.py \
    --split test \
    --locked-coefs-from ../results/best_coefs_tune.json
# ~1 h on H100

# ---- paper artefacts (CPU only) ----
python ../paper/scripts/build_all.py
# Writes paper/figures/*.{pdf,png} at 300 DPI and paper/tables/*.{csv,tex}
```

Total GPU budget: **~26–34 H100 hours**, approximately **$65–85** at
Lambda on-demand H100 rates.

Trace paths: every numerical claim in `paper/RESULTS.md` cites the exact
`results/*.json` or `paper/tables/*.csv` file it is drawn from.

---

## Table of Contents

1. [What this experiment does](#what-this-experiment-does)
2. [Prerequisites](#prerequisites)
3. [Lambda GPU setup (step by step)](#lambda-gpu-setup)
4. [Running the experiment](#running-the-experiment)
5. [Claude Code prompt](#claude-code-prompt-for-running-on-lambda)
6. [Cost and time estimates](#cost-and-time-estimates)
7. [Troubleshooting](#troubleshooting)
8. [Expected outputs](#expected-outputs)
9. [Repo layout](#repo-layout)
10. [Key design decisions](#key-design-decisions)

---

## What this experiment does

### The question

Can general persona vectors (skeptic, judge, devil's advocate, etc.) reduce
sycophancy as effectively as a targeted sycophancy-specific vector (CAA)?
And if so, how does each vector reduce it -- through healthy disagreement,
neutral hedging, or model degradation?

### The five parts

| Part | What it does | Script(s) | GPU time |
|------|-------------|-----------|----------|
| 1. Steering evaluation | Sweep 21 conditions x 13 coefficients on 600 eval rows | `02_evaluate_steering.py` | ~6-8 hrs |
| 2. CAA extraction | Extract targeted sycophancy vector (Rimsky et al.) | `02b_extract_caa.py` | ~30 min |
| 3. Geometric analysis | Cosine similarity matrix, PCA persona space | `01` + `03` | ~1 min |
| 4. Decomposition | Split each role into CAA-aligned + residual | `01` + `03` | ~1 min |
| 5. Mechanistic | Axis projection predicts sycophancy, response-token probes | `02` + `03` | included in Part 1 |

### The 21 steering conditions

| Category | Conditions | Expected effect |
|----------|-----------|-----------------|
| Broad axis | assistant_axis | Decrease sycophancy (toward grounded assistant) |
| Critical roles | skeptic, judge, devils_advocate, contrarian, scientist | Decrease sycophancy (toward independent thinking) |
| Conformist roles | servant, diplomat, disciple, peacemaker | **Increase** sycophancy (toward deference) |
| Targeted baseline | caa (Contrastive Activation Addition) | Decrease sycophancy (sycophancy-specific vector) |
| Controls | random_0 through random_9 | No directional effect (noise) |

---

## Prerequisites

### Hardware
- **GPU**: 1x H100 80GB or 1x A100 80GB (~55 GB VRAM needed for Gemma 2 27B bf16)
- **Disk**: ~50 GB free (model weights ~15GB + HF cache + vectors + results)
- **RAM**: 64 GB+ system RAM recommended

### Accounts
- **Lambda Cloud account** with GPU credits (~$35-50 needed for full experiment)
- **HuggingFace account** with:
  - An access token (create at https://huggingface.co/settings/tokens)
  - Access granted to `google/gemma-2-27b-it` (request at the model page -- approval is usually instant)

### Software (pre-installed on Lambda)
- Python 3.10+
- CUDA 12+
- PyTorch (pre-installed on Lambda GPU instances)

---

## Lambda GPU setup

### Step 0: Launch a Lambda instance

1. Go to https://cloud.lambdalabs.com/instances
2. Launch a **1x H100 80GB SXM** instance (cheapest option that works)
   - If H100 is unavailable, **1x A100 80GB** also works
   - Do NOT use A100 40GB -- not enough VRAM
3. Wait for it to boot, copy the SSH command

### Step 1: SSH in and use tmux

**CRITICAL**: Always work inside tmux. The experiment takes hours. If your
SSH connection drops without tmux, you lose everything.

```bash
ssh ubuntu@<your-lambda-ip>

# Start a tmux session (or reattach if you got disconnected)
tmux new-session -s experiment
# If reconnecting later: tmux attach -t experiment
```

### Step 2: Clone the repo and install dependencies

```bash
cd /home/ubuntu
git clone <your-repo-url> experiment
cd experiment

# Install Python dependencies (torch is already on Lambda)
pip install transformers accelerate huggingface_hub datasets \
            scipy scikit-learn statsmodels matplotlib seaborn
```

### Step 3: HuggingFace authentication

```bash
# Login to HuggingFace (paste your token when prompted)
huggingface-cli login

# Verify you have access to Gemma 2 27B
python -c "from huggingface_hub import model_info; m = model_info('google/gemma-2-27b-it'); print(f'OK: {m.id}')"
```

If the verify command fails with a 403 error, you need to:
1. Go to https://huggingface.co/google/gemma-2-27b-it
2. Click "Agree and access" (Google's license agreement)
3. Wait a few minutes, try again

### Step 4: Clone the steering library

```bash
git clone https://github.com/safety-research/assistant-axis.git /home/ubuntu/assistant-axis
```

### Step 5: Set environment variables

```bash
# Add to ~/.bashrc so they persist across tmux panes
echo 'export EXPERIMENT_ROOT=/home/ubuntu/experiment' >> ~/.bashrc
echo 'export ASSISTANT_AXIS_PATH=/home/ubuntu/assistant-axis' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Verify the setup

```bash
cd /home/ubuntu/experiment/scripts

# Quick sanity check -- should print all config values without errors
python -c "from config import *; print(f'ROOT={ROOT}'); print(f'MODEL={MODEL_NAME}'); print(f'CONDITIONS={len(CONDITIONS)} total'); print('Config OK')"
```

You should see:
```
ROOT=/home/ubuntu/experiment
MODEL=google/gemma-2-27b-it
CONDITIONS=21 total
Config OK
```

---

## Running the experiment

**All commands below assume you are in `/home/ubuntu/experiment/scripts/` inside tmux.**

### Phase 1: Data setup (~15 minutes, loads model once for smoke test)

```bash
cd /home/ubuntu/experiment/scripts

# 1a. Download vectors + datasets, run sanity check
#     This loads the model, so it takes ~10 min on first run (downloading weights)
#     Subsequent runs are fast because weights are cached in ~/.cache/huggingface/
python 00_setup.py

# 1b. Download the REAL philpapers2020 data
#     (The HuggingFace mirror has the wrong content under this filename)
curl -sLk "https://api.github.com/repos/anthropics/evals/git/blobs/5525210614d4f26b1732042e7dcb7210d23fe5aa" \
     -H "Accept: application/vnd.github.raw" \
     -o ../data/sycophancy/sycophancy_on_philpapers2020.jsonl

# 1c. Verify the sycophancy datasets exist
echo "--- Checking datasets ---"
wc -l ../data/sycophancy/sycophancy_on_philpapers2020.jsonl
wc -l ../data/sycophancy/sycophancy_on_nlp_survey.jsonl
wc -l ../data/sycophancy/sycophancy_on_political_typology.jsonl
# All three should show line counts (not "No such file")

# 1d. Build the counterbalanced eval set (300 base x 2 = 600 rows)
python 00b_rebuild_eval.py
```

**Checkpoint**: You should see `Wrote /home/ubuntu/experiment/data/eval_data.json` with 600 rows.

### Phase 2: Vector preparation (~30 min for CAA, ~1 min for roles)

```bash
# 2a. Extract CAA sycophancy vector
#     This loads the model and runs ~800 paired forward passes
#     Takes ~30 min on H100
python 02b_extract_caa.py

# 2b. Prepare all steering vectors + decomposition
#     This is fast (just tensor math, no model needed)
python 01_prepare_steering_vectors.py
```

**Checkpoint**: You should see:
- `vectors/steering/caa_unit.pt` exists
- `Cosine similarity(CAA, assistant_axis) = ...` printed
- Decomposition table with cosine values for each role

### Phase 3: Main evaluation (~6-8 hours)

This is the expensive part. Use `--resume` if you need to restart.

**RECOMMENDED: Full run (simplest, ~6-8 hours)**
```bash
# Run all 21 conditions x 13 coefficients on all 600 eval rows
# Progress is printed per condition. Checkpoints saved per condition.
python 02_evaluate_steering.py 2>&1 | tee ../results/eval_log.txt

# If interrupted (SSH drop, OOM, etc.), resume from checkpoints:
# python 02_evaluate_steering.py --resume
```

**ALTERNATIVE: Tune/test split (stronger for publication, ~7-8 hours total)**
```bash
# Run on tune split first (~3-4 hours)
python 02_evaluate_steering.py --split tune 2>&1 | tee ../results/eval_tune_log.txt

# Quick analysis on tune split to verify things look sane
python 03_analysis.py --split tune

# Then run on held-out test split (~3-4 hours)
python 02_evaluate_steering.py --split test 2>&1 | tee ../results/eval_test_log.txt
```

**SMOKE TEST (5 minutes, to verify everything works before committing to full run)**
```bash
python 02_evaluate_steering.py --max-base 5
# This runs only 5 base questions (10 rows) -- fast but shows if the pipeline works
```

**Checkpoint**: After completion:
- `results/all_results.json` should exist (or `all_results_tune.json` / `all_results_test.json`)
- `results/sycophancy_rates.json` should exist
- The eval log should show per-condition sycophancy rates

### Phase 4: Analysis (~5 minutes, CPU only)

```bash
# Generate all figures + statistical tests + summary
python 03_analysis.py
# Or if you used tune/test split:
# python 03_analysis.py --split test

# View the summary
cat ../results/summary.txt
```

**Checkpoint**: `figures/` should contain fig1 through fig6. `results/summary.txt` should show the results table.

### Phase 5: Qualitative evaluation (~30-45 min, loads model)

```bash
# Structured over-correction eval (16 true/false probes, 5 conditions)
python over_correction_check.py 2>&1 | tee ../results/overcorrection_log.txt

# Free-form generation comparison (5 sycophancy prompts, 5 conditions)
python qual_check.py 2>&1 | tee ../results/qualcheck_log.txt
```

### Phase 6: Download results

From your local machine:

```bash
# Download all results and figures
scp -r ubuntu@<lambda-ip>:/home/ubuntu/experiment/results ./results_from_lambda/
scp -r ubuntu@<lambda-ip>:/home/ubuntu/experiment/figures ./figures_from_lambda/
```

---

## Claude Code prompt for running on Lambda

If your friend is using Claude Code on the Lambda instance, paste this prompt
to have Claude run the experiment:

```
I need you to run a sycophancy steering experiment on this Lambda GPU.
The codebase is at /home/ubuntu/experiment and the steering library is
at /home/ubuntu/assistant-axis.

IMPORTANT CONTEXT:
- The model is Gemma 2 27B (google/gemma-2-27b-it), needs ~55GB VRAM
- I am already SSH'd into a Lambda instance with 1x H100 80GB
- HuggingFace is already authenticated (huggingface-cli login done)
- The assistant-axis repo is already cloned at /home/ubuntu/assistant-axis
- Environment variables EXPERIMENT_ROOT and ASSISTANT_AXIS_PATH are set

WHAT TO DO:
Run the following scripts IN ORDER from /home/ubuntu/experiment/scripts/.
Each script depends on the previous one. Do NOT skip any step.
Do NOT run steps in parallel -- they must be sequential.

Step 1: python 00_setup.py
Step 2: curl the philpapers blob (see README for the exact curl command)
Step 3: python 00b_rebuild_eval.py
Step 4: python 02b_extract_caa.py
Step 5: python 01_prepare_steering_vectors.py
Step 6: python 02_evaluate_steering.py --max-base 5
        (this is a smoke test -- verify it works before the full run)
Step 7: python 02_evaluate_steering.py
        (this is the full run, takes ~6-8 hours -- use --resume if interrupted)
Step 8: python 03_analysis.py
Step 9: python over_correction_check.py
Step 10: python qual_check.py

After each step, check:
- Step 1: data/eval_data_raw.json exists
- Step 3: data/eval_data.json exists with 600 entries
- Step 4: vectors/steering/caa_unit.pt exists
- Step 5: vectors/steering/ has ~25 .pt files
- Step 6: prints sycophancy rates for 5 base questions (smoke test)
- Step 7: results/all_results.json exists (this is the big one)
- Step 8: figures/ has fig1 through fig6, results/summary.txt exists
- Steps 9-10: results/over_correction_eval.json exists

If any step fails, read the error carefully. Common issues:
- "CUDA out of memory" -> the GPU is too small or another process is using it
  Run: nvidia-smi to check. Kill other processes if needed.
- "FileNotFoundError" for eval_data.json -> you skipped step 3 (00b)
- "KeyError: 'base_id'" -> you ran 02 before 00b (the schema check should catch this)
- "401 or 403 from HuggingFace" -> token issue, re-run huggingface-cli login

The most expensive step is Step 7 (~6-8 hours). Everything else is fast.
Do the smoke test (Step 6) first to catch any config/auth issues early.

After completion, tell me what's in results/summary.txt.
```

---

## Cost and time estimates

Assuming **Lambda 1x H100 80GB at ~$2.49/hr** (on-demand pricing as of early 2026):

| Phase | What | GPU hours | Cost |
|-------|------|-----------|------|
| Setup (00, 00b) | Download model + vectors, sanity check, build eval set | ~0.3 hrs | ~$0.75 |
| CAA extraction (02b) | ~800 paired forward passes | ~0.5 hrs | ~$1.25 |
| Vector prep (01) | Tensor math, no model needed | ~0.01 hrs | ~$0.02 |
| **Main eval (02)** | **21 conditions x 12 non-zero coefficients x 600 rows** | **~6-8 hrs** | **~$15-20** |
| Analysis (03) | CPU-only figures and stats | ~0.05 hrs | ~$0.12 |
| Qualitative (over_correction, qual_check) | Generation under 5 conditions | ~0.75 hrs | ~$1.87 |
| **Total** | | **~8-10 hrs** | **~$20-25** |

### How to minimize costs

1. **Always use tmux** so SSH disconnects don't kill runs.
2. **Run the smoke test first** (`--max-base 5`) to catch auth/config issues before committing to the full run. The smoke test costs ~$0.10.
3. **Use `--resume`** if a run is interrupted -- it reads checkpoint files and skips completed conditions.
4. **Don't re-download the model** -- once cached in `~/.cache/huggingface/`, subsequent runs load from cache (~2 min vs ~10 min).
5. **Stop the instance when done** -- Lambda charges by the hour even when idle.

### What NOT to do (common mistakes that waste money)

- Do NOT run `02_evaluate_steering.py` without first running `00b_rebuild_eval.py` -- it will crash, wasting the model loading time (~$0.50).
- Do NOT run multiple instances of `02_evaluate_steering.py` in parallel on the same GPU -- they will OOM.
- Do NOT use `--max-base` for the real run -- that's for smoke tests only.
- Do NOT forget to `--resume` after an interruption -- without it, the script starts over from scratch.

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Check what's using the GPU
nvidia-smi
# Kill any stray Python processes
pkill -f "python.*evaluate"
# Then retry
```

### "FileNotFoundError: eval_data.json"
You skipped `00b_rebuild_eval.py`. Run it first.

### "KeyError: 'base_id'" or "KeyError: 'variant'"
The script printed a clear error message telling you to run `00b_rebuild_eval.py`.
This means `eval_data.json` exists but is the old format (from `00_setup.py`).
Run `00b_rebuild_eval.py` to produce the counterbalanced version.

### "ModuleNotFoundError: assistant_axis"
The `ASSISTANT_AXIS_PATH` environment variable is not set, or the repo isn't cloned.
```bash
export ASSISTANT_AXIS_PATH=/home/ubuntu/assistant-axis
# Check it exists:
ls /home/ubuntu/assistant-axis/assistant_axis/steering.py
```

### "401 Client Error" or "403 Forbidden" from HuggingFace
```bash
# Re-authenticate
huggingface-cli login
# Verify access
python -c "from huggingface_hub import model_info; print(model_info('google/gemma-2-27b-it').id)"
```
If you still get 403, go to https://huggingface.co/google/gemma-2-27b-it and accept the license.

### "Cannot import config" or "ModuleNotFoundError: config"
You're running from the wrong directory. All scripts must be run from `scripts/`:
```bash
cd /home/ubuntu/experiment/scripts
python 02_evaluate_steering.py
```

### The run was interrupted -- how do I resume?
```bash
python 02_evaluate_steering.py --resume
```
This reads checkpoint files from `results/checkpoints/` and skips conditions that are already complete.

### SSH disconnected mid-run
If you used tmux (as instructed), just reconnect:
```bash
ssh ubuntu@<lambda-ip>
tmux attach -t experiment
```
Your run is still going. If you didn't use tmux, the process died. Use `--resume` to restart.

### Results look wrong (all conditions show same rate)
This usually means the model isn't actually being steered. Check:
1. Does `vectors/steering/skeptic_unit.pt` exist and have the right shape?
   ```bash
   python -c "import torch; v = torch.load('../vectors/steering/skeptic_unit.pt', weights_only=False); print(v.shape, v.norm())"
   ```
   Should print `torch.Size([4608])` and norm ~1.0.
2. Is the target layer correct? Should be 22 for Gemma 2 27B.

### I want to re-run analysis without re-running the eval
```bash
# Analysis is CPU-only and reads from results/all_results.json
python 03_analysis.py
# This takes ~5 minutes and overwrites figures/ and results/summary.txt
```

---

## Expected outputs

### After the full pipeline

```
results/
  summary.txt                       <- START HERE: human-readable results
  summary_tune.txt                  <- tune-split results (if using tune/test)
  summary_test.txt                  <- test-split results (reportable numbers)
  statistical_tests.json            <- all tests with raw + adjusted p-values
  sycophancy_rates.json             <- per-(condition, coefficient) rates and logits
  degradation_flags.json            <- which cells are degraded
  vector_cosine_similarities.json   <- pairwise cosine sim between all vectors
  over_correction_eval.json         <- structured probe results (16 claims x 5 conditions)
  tune_test_split.json              <- which base_ids went to tune vs test
  all_results.json                  <- per-row raw data (~16 MB, all 21x13x600 rows)
  checkpoints/                      <- per-condition checkpoints (for --resume)

figures/
  fig1_steering_curves.png          <- THE MAIN FIGURE: sycophancy vs coefficient
  fig2_projection_distributions.png <- axis projection for sycophantic vs honest
  fig3_projection_shift.png         <- how steering shifts the projection
  fig4_best_comparison.png          <- best per condition, bootstrap 95% CI
  fig5_vector_similarities.png      <- cosine similarity heatmap
  fig6_persona_space_pca.png        <- PCA of steering vectors in persona space
```

### What to look for in the results

In `summary.txt`:
- **Baseline sycophancy rate** should be ~60% (if much higher, counterbalancing may have failed)
- **Critical roles** should show logit LOWER than baseline (reducing sycophancy)
- **Conformist roles** should show logit HIGHER than baseline (increasing sycophancy)
- **CAA** should show strong reduction (it's the targeted vector)
- **Random** should show no consistent direction (noise control)
- **Wilcoxon p_adj** should be significant (< 0.05) for the best conditions

---

## Repo layout

```
.
├── README.md                           <- you are here
├── scripts/
│   ├── config.py                       # centralized constants (paths, model, roles, etc.)
│   ├── 00_setup.py                     # [STEP 1] download vectors + datasets, sanity check
│   ├── 00b_rebuild_eval.py             # [STEP 3] counterbalanced eval set (--seed for multi-seed)
│   ├── 02b_extract_caa.py              # [STEP 4] CAA vector extraction (Rimsky et al.)
│   ├── 01_prepare_steering_vectors.py  # [STEP 5] all vectors + decomposition + cosine matrix
│   ├── 02_evaluate_steering.py         # [STEP 7] main eval (--split, --resume, --max-base)
│   ├── 03_analysis.py                  # [STEP 8] figures + stats + decomposition + PCA
│   ├── over_correction_check.py        # [STEP 9] structured ground-truth probes
│   ├── qual_check.py                   # [STEP 10] free-form generation comparison
│   ├── find_skeptic_agreement.py       # where skeptic agrees with user
│   ├── pilot_debug.py                  # verify hooks and sign convention
│   ├── pilot_scale.py                  # find coefficient regime
│   └── run_multiseed.py                # multi-seed experiment runner
│
├── vectors/steering/                   # unit-normalized vectors (produced by scripts)
├── data/                               # eval datasets
├── results/                            # all outputs (figures, stats, summaries)
└── figures/                            # generated figures (png)
```

---

## Key design decisions

### Counterbalancing
Gemma 2 27B has a ~93% positional A-bias. Every base question is evaluated
in both A/B orderings. The A-bias cancels in the mean.

### Tune/test split
Best coefficient selection and final reporting use different data splits
(at the base-question level, so counterbalanced pairs stay together).
This prevents selection bias (the #1 issue in the original experiment design).

### 10 random controls
Random vectors are averaged to produce a stable control baseline with
uncertainty bands, rather than relying on a single random draw.

### Bidirectional control
Critical roles (skeptic, judge, etc.) are expected to decrease sycophancy.
Conformist roles (servant, diplomat, etc.) are expected to increase it.
If both directions work, persona steering genuinely controls the sycophancy
axis -- it's not just noise.

### CAA comparison
The CAA vector (Rimsky et al. 2024) is the targeted sycophancy-specific
baseline. It is extracted from different datasets (NLP survey + political
typology) than the eval set (philpapers2020), ensuring no data leakage.
The decomposition analysis reveals whether role vectors work by overlapping
with the CAA direction or through independent mechanisms.

### Statistical framework
- **Primary test**: Paired Wilcoxon signed-rank on base-level syc_logit
- **Multiple comparison correction**: Holm-Bonferroni across all test families
- **Dose-response**: Row-level Spearman (not 6-point aggregate)
- **McNemar**: exact test when off-diagonal cells < 5
- **p-value underflow**: guarded (never reports 0.0)
- **Degradation detection**: compares to random-mean at matched coefficient

### Sign convention
- **Assistant Axis**: positive coefficient pushes TOWARD assistant (more grounded)
- **Role directions**: positive coefficient pushes TOWARD the role (e.g., more skeptical)
- **CAA vector**: points FROM honest TOWARD sycophantic; use NEGATIVE coefficient to reduce sycophancy
