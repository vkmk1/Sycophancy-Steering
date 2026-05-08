"""Reconstruct per-condition checkpoint files from all_results_{tune,test}.json.

02_evaluate_steering.py --resume needs results/checkpoints/{cond}{suffix}.json
and results/checkpoints/baseline{suffix}.json. The repo ships the aggregated
all_results_*.json but not the individual checkpoints, so we rebuild them here
from the existing 21-condition data. This lets a subsequent --resume run only
evaluate the new residual_standalone conditions on the GPU.
"""
import json
import os
from collections import defaultdict

import os as _os
ROOT = _os.environ.get(
    "EXPERIMENT_ROOT", "/lambda/nfs/filesystem/sycophancy-final/experiment-main"
)


def reconstruct(split: str):
    suffix = f"_{split}"
    src = f"{ROOT}/results/all_results{suffix}.json"
    if not os.path.exists(src):
        print(f"[skip] {src} not present")
        return
    with open(src) as f:
        rows = json.load(f)
    ck_dir = f"{ROOT}/results/checkpoints"
    os.makedirs(ck_dir, exist_ok=True)

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    # Each condition's checkpoint in 02_evaluate_steering.py is the list of
    # NON-ZERO-coef rows for that condition. Baseline is the shared coef=0 rows
    # (stored once under condition "baseline").
    first_cond = next(iter(by_cond))
    baseline_rows = [dict(r) for r in by_cond[first_cond] if r["coefficient"] == 0.0]
    for br in baseline_rows:
        br["condition"] = "baseline"
    with open(f"{ck_dir}/baseline{suffix}.json", "w") as f:
        json.dump(baseline_rows, f)
    print(f"wrote baseline{suffix}.json ({len(baseline_rows)} rows)")

    for cond, cr in by_cond.items():
        nonzero = [r for r in cr if r["coefficient"] != 0.0]
        with open(f"{ck_dir}/{cond}{suffix}.json", "w") as f:
            json.dump(nonzero, f)
        print(f"wrote {cond}{suffix}.json ({len(nonzero)} rows)")


if __name__ == "__main__":
    for split in ("tune", "test"):
        reconstruct(split)
