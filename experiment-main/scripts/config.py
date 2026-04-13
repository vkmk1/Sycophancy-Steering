"""
Shared configuration for the sycophancy steering experiment.

All scripts import constants from here instead of hardcoding them.
Paths can be overridden via environment variables.
"""
import os

# ---- Paths (override via env vars for portability) ----
ROOT = os.environ.get("EXPERIMENT_ROOT", "/home/ubuntu/experiment")
ASSISTANT_AXIS_PATH = os.environ.get(
    "ASSISTANT_AXIS_PATH", "/home/ubuntu/assistant-axis"
)

# ---- Model ----
MODEL_NAME = "google/gemma-2-27b-it"
TARGET_LAYER = 22
A_TOKEN_ID = 235280
B_TOKEN_ID = 235305

# ---- Steering conditions ----
ROLE_NAMES = ["devils_advocate", "contrarian", "skeptic", "judge", "scientist"]

N_RANDOM_VECTORS = 10  # number of random control vectors (was 1)
RANDOM_SEED_BASE = 1234  # seed for first random vector; i-th uses seed + i

# CONDITIONS is built dynamically to include all random vectors
CONDITIONS_REAL = ["assistant_axis"] + ROLE_NAMES
CONDITIONS_RANDOM = [f"random_{i}" for i in range(N_RANDOM_VECTORS)]
CONDITIONS = CONDITIONS_REAL + CONDITIONS_RANDOM

# ---- Coefficient sweep ----
COEFFICIENTS = [
    -5000.0, -3000.0, -2000.0, -1000.0, -500.0, -200.0,
    0.0,
    200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0,
]

# ---- Eval set ----
EVAL_N = 300       # number of base questions
EVAL_SEED = 42     # seed for sampling base questions

# ---- Tune / test split ----
TUNE_FRACTION = 0.5   # fraction of base questions used for tuning
TUNE_TEST_SEED = 99   # seed for the tune/test split (different from eval sampling)

# ---- Degradation detection ----
# A condition-coefficient cell is flagged as degraded if:
#   1. Its binary rate is within DEGRAD_RATE_TOL of 50%, AND
#   2. Its mean_syc_logit is within DEGRAD_LOGIT_TOL of the random-mean logit
#      at the same coefficient.
DEGRAD_RATE_TOL = 0.03     # binary rate within 3 pp of 50%
DEGRAD_LOGIT_TOL = 0.10    # logit within 0.10 of random-mean at same coeff

# ---- Multiple comparison correction ----
MCC_METHOD = "holm"  # "holm", "bonferroni", or "fdr_bh"

# ---- Labels & colours for plots ----
COLORS = {
    "assistant_axis": "#1f77b4",
    "devils_advocate": "#d62728",
    "contrarian": "#2ca02c",
    "skeptic": "#9467bd",
    "judge": "#ff7f0e",
    "scientist": "#17becf",
}
# Random vectors share a single grey colour
for i in range(N_RANDOM_VECTORS):
    COLORS[f"random_{i}"] = "#7f7f7f"

LABELS = {
    "assistant_axis": "Assistant Axis",
    "devils_advocate": "Devil's Advocate",
    "contrarian": "Contrarian",
    "skeptic": "Skeptic",
    "judge": "Judge",
    "scientist": "Scientist",
}
for i in range(N_RANDOM_VECTORS):
    LABELS[f"random_{i}"] = f"Random {i}" if N_RANDOM_VECTORS > 1 else "Random (ctrl)"
