"""
Shared configuration for the sycophancy steering experiment.

All scripts import constants from here instead of hardcoding them.
Paths can be overridden via environment variables.
"""
import os

# ---- Paths (override via env vars for portability) ----
ROOT = os.environ.get("EXPERIMENT_ROOT", "/lambda/nfs/filesystem/sycophancy-qwen")
ASSISTANT_AXIS_PATH = os.environ.get(
    "ASSISTANT_AXIS_PATH", "/home/ubuntu/assistant-axis"
)

# ---- Model (Qwen 3 32B replication) ----
# 64 decoder layers, hidden_size 5120. TARGET_LAYER is the canonical layer
# chosen by the assistant-axis authors for this model (see
# assistant_axis/models.py MODEL_CONFIGS['Qwen/Qwen3-32B']['target_layer'] = 32).
# Answer-token IDs are Qwen tokenizer single-token IDs for "A" and "B"
# (verified: tok.encode('A', add_special_tokens=False) -> [32]).
MODEL_NAME = "Qwen/Qwen3-32B"
TARGET_LAYER = 32
A_TOKEN_ID = 32
B_TOKEN_ID = 33

# ---- Steering conditions ----
# Critical roles: expected to DECREASE sycophancy (push toward independent thinking)
CRITICAL_ROLES = ["devils_advocate", "contrarian", "skeptic", "judge", "scientist"]

# Conformist roles: expected to INCREASE sycophancy (push toward deference).
# Tests bidirectionality: if critical roles reduce syc, conformist roles should increase it.
#
# The original list ("servant", "diplomat", "disciple", "peacemaker") was not
# available in lu-christina/assistant-axis-vectors, so we substitute the closest
# conflict-avoidant / agreement-seeking roles that ARE pre-computed in that
# dataset. All four are defensible conformist personas (peacekeeper avoids
# conflict, pacifist avoids confrontation, collaborator agrees to work together,
# facilitator enables others without pushing back). Revision work could generate
# the original four via the safety-research/assistant-axis pipeline, but that
# costs ~4-8 extra H100 hours and is out of scope for the first submission.
CONFORMIST_ROLES = ["peacekeeper", "pacifist", "collaborator", "facilitator"]

ROLE_NAMES = CRITICAL_ROLES + CONFORMIST_ROLES

N_RANDOM_VECTORS = 10  # number of random control vectors
RANDOM_SEED_BASE = 1234  # seed for first random vector; i-th uses seed + i

# Standalone CAA-orthogonal residuals. Each is the role vector minus its
# projection onto CAA, unit-normalized. Evaluated as fresh independent
# steering conditions with their own tune-locked coefficients.
#   skeptic, contrarian: representative low-|cos(CAA)| residuals (matches
#     the Gemma replication; lets us compare like-for-like across models).
#   devils_advocate: highest-|cos(CAA)| role on Qwen 3 32B (0.108) using
#     the OFFICIAL lu-christina/assistant-axis-vectors/qwen-3-32b release
#     at layer 32. On Gemma the highest was collaborator at 0.165; on Qwen
#     the ordering is different -- devils_advocate edges out skeptic
#     (0.108 vs 0.105), with facilitator in 4th at 0.056. We apply the
#     user's "pick whichever critical or conformist role has the highest
#     |cos(role, CAA)|" rule per model.
RESIDUAL_STANDALONE_ROLES = ["skeptic", "contrarian", "devils_advocate"]
CONDITIONS_RESIDUAL_STANDALONE = [f"{r}_residual" for r in RESIDUAL_STANDALONE_ROLES]

# CONDITIONS is built dynamically to include all conditions
CONDITIONS_REAL = (
    ["assistant_axis"] + ROLE_NAMES + ["caa"] + CONDITIONS_RESIDUAL_STANDALONE
)
CONDITIONS_CRITICAL = ["assistant_axis"] + CRITICAL_ROLES + ["caa"]
CONDITIONS_CONFORMIST = CONFORMIST_ROLES
CONDITIONS_RANDOM = [f"random_{i}" for i in range(N_RANDOM_VECTORS)]
CONDITIONS = CONDITIONS_REAL + CONDITIONS_RANDOM

# Type tag used when emitting rows to the results CSV so the residual-standalone
# rows can be filtered out of primary-result tables and into the ablation table.
CONDITION_TYPE = {}
for _c in ["assistant_axis", "caa"]:
    CONDITION_TYPE[_c] = "primary"
for _r in CRITICAL_ROLES:
    CONDITION_TYPE[_r] = "critical_role"
for _r in CONFORMIST_ROLES:
    CONDITION_TYPE[_r] = "conformist_role"
for _c in CONDITIONS_RANDOM:
    CONDITION_TYPE[_c] = "random_control"
for _c in CONDITIONS_RESIDUAL_STANDALONE:
    CONDITION_TYPE[_c] = "residual_standalone"

# ---- Coefficient sweep ----
# NB: Qwen 3 32B's layer-31 activations have much smaller natural norm than
# Gemma-2-27b-it's layer-22 activations (default_mean_norm ≈ 156 on Qwen
# vs. an order of magnitude larger on Gemma). Steering along a unit vector
# with coef = 5000 on Qwen completely destroys the next-token distribution
# (top-1 becomes single whitespace / '<|im_start|>' tokens). We rescaled
# the grid to [-500, -200, -100, -50, 0, 50, 100, 200, 500], which on a
# pilot probe keeps the top-1 token recognisable as 'Answer'/'B'/'A' while
# still traversing from clearly sycophantic (low |coef|) to
# anti-sycophantic (~coef=200) to saturated (>=500).
COEFFICIENTS = [
    -500.0, -200.0, -100.0, -50.0,
    0.0,
    50.0, 100.0, 200.0, 500.0,
]
COEFFICIENTS_FULL = [
    -500.0, -300.0, -200.0, -100.0, -50.0, -20.0,
    0.0,
    20.0, 50.0, 100.0, 200.0, 300.0, 500.0,
]

# ---- Eval set ----
EVAL_N = 300       # number of base questions
EVAL_SEED = 42     # seed for sampling base questions

# ---- CAA extraction ----
# NLP survey + political typology are used for CAA extraction (training set).
# philpapers2020 is the held-out test set.
# NOTE: the real filename in anthropics/evals is *_political_typology_quiz.jsonl,
# not *_political_typology.jsonl. The earlier filename silently halved the
# CAA training set because 02b_extract_caa.py warned-and-continued.
CAA_DATASETS = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]
EVAL_DATASET = "sycophancy_on_philpapers2020.jsonl"

# ---- Tune / test split ----
TUNE_FRACTION = 0.5   # fraction of base questions used for tuning
TUNE_TEST_SEED = 99   # seed for the tune/test split (different from eval sampling)

# ---- Degradation detection ----
DEGRAD_RATE_TOL = 0.03     # binary rate within 3 pp of 50%
DEGRAD_LOGIT_TOL = 0.10    # logit within 0.10 of random-mean at same coeff

# ---- Multiple comparison correction ----
MCC_METHOD = "holm"  # "holm", "bonferroni", or "fdr_bh"

# ---- Labels & colours for plots ----
COLORS = {
    "assistant_axis": "#1f77b4",
    "caa":            "#e377c2",
    # Critical roles
    "devils_advocate": "#d62728",
    "contrarian": "#2ca02c",
    "skeptic": "#9467bd",
    "judge": "#ff7f0e",
    "scientist": "#17becf",
    # Conformist roles
    "peacekeeper": "#8c564b",
    "pacifist":    "#bcbd22",
    "collaborator": "#e7969c",
    "facilitator": "#aec7e8",
    # Standalone CAA-orthogonal residuals (distinct hues so they read as a
    # separate family from their parent roles in fig1/fig4).
    "skeptic_residual":         "#6a3d9a",
    "contrarian_residual":      "#1b9e77",
    "collaborator_residual":    "#b15928",
    "facilitator_residual":     "#b15928",
    "devils_advocate_residual": "#b15928",
}
for i in range(N_RANDOM_VECTORS):
    COLORS[f"random_{i}"] = "#7f7f7f"

LABELS = {
    "assistant_axis": "Assistant Axis",
    "caa":            "CAA (targeted)",
    # Critical roles
    "devils_advocate": "Devil's Advocate",
    "contrarian": "Contrarian",
    "skeptic": "Skeptic",
    "judge": "Judge",
    "scientist": "Scientist",
    # Conformist roles
    "peacekeeper": "Peacekeeper",
    "pacifist":    "Pacifist",
    "collaborator": "Collaborator",
    "facilitator": "Facilitator",
    # Standalone residuals
    "skeptic_residual":         "Skeptic \u22a5 CAA",
    "contrarian_residual":      "Contrarian \u22a5 CAA",
    "collaborator_residual":    "Collaborator \u22a5 CAA",
    "facilitator_residual":     "Facilitator \u22a5 CAA",
    "devils_advocate_residual": "Devil's Advocate \u22a5 CAA",
}
for i in range(N_RANDOM_VECTORS):
    LABELS[f"random_{i}"] = f"Random {i}" if N_RANDOM_VECTORS > 1 else "Random (ctrl)"


# ---- Expected direction of effect per condition ----
# Used by the direction-aware best-coefficient selector in 03_analysis.py and by
# paper/scripts/* so that conformist roles are not evaluated with the wrong
# operating point. See GAP_ANALYSIS.md Objective 1, item 1.
#
#   "decrease": steering this vector should reduce sycophancy (lower syc_logit).
#               Applies to assistant_axis, critical roles, and CAA (CAA points
#               FROM honest TOWARD sycophantic, so a negative coefficient
#               reduces sycophancy -- the selector handles the sign).
#   "increase": steering this vector should raise sycophancy (bidirectionality
#               check -- conformist roles).
#   "unsigned": no prior; pick the coefficient with the largest absolute delta
#               from the random mean (random controls).
EXPECTED_DIRECTION = {
    "assistant_axis": "decrease",
    "caa":            "decrease",
}
for _r in CRITICAL_ROLES:
    EXPECTED_DIRECTION[_r] = "decrease"
for _r in CONFORMIST_ROLES:
    EXPECTED_DIRECTION[_r] = "increase"
for _i in range(N_RANDOM_VECTORS):
    EXPECTED_DIRECTION[f"random_{_i}"] = "unsigned"
# Standalone residuals all use the "decrease" selector. Rationale: the
# hypothesis under test is the decomposition claim that low-|cos(CAA)|
# residuals reduce sycophancy while high-|cos(CAA)| residuals do not. That
# is a statement about syc REDUCTION for all three residuals, irrespective
# of whether their parent role was critical or conformist. Using parent
# direction (e.g. "increase" for collaborator_residual) would test a
# different, tangential question.
for _role in RESIDUAL_STANDALONE_ROLES:
    EXPECTED_DIRECTION[f"{_role}_residual"] = "decrease"
