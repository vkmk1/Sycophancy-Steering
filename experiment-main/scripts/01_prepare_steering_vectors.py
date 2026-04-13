"""
Script 1: Prepare steering vectors.

- Extract axis @ target layer (already a direction: default - role_mean)
- For each role: direction = role - default, then unit-normalize
- Generate N_RANDOM_VECTORS random unit control vectors (same dtype/shape)
- Write all normalized vectors + pairwise cosine similarity matrix
"""
import json
import os

import torch

from config import (
    ROOT, TARGET_LAYER, ROLE_NAMES,
    N_RANDOM_VECTORS, RANDOM_SEED_BASE,
)

os.makedirs(f"{ROOT}/vectors/steering", exist_ok=True)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def main():
    # Load raw tensors as float32 for numerics
    axis = torch.load(
        f"{ROOT}/vectors/gemma-2-27b/assistant_axis.pt",
        map_location="cpu",
        weights_only=False,
    ).float()
    default = torch.load(
        f"{ROOT}/vectors/gemma-2-27b/default_vector.pt",
        map_location="cpu",
        weights_only=False,
    ).float()

    axis_vec = axis[TARGET_LAYER]  # shape [hidden_dim]
    default_vec = default[TARGET_LAYER]
    print(
        f"axis @ L{TARGET_LAYER}: shape={tuple(axis_vec.shape)}  norm={axis_vec.norm():.4f}"
    )
    print(
        f"default @ L{TARGET_LAYER}: shape={tuple(default_vec.shape)}  norm={default_vec.norm():.4f}"
    )

    prepared = {}
    prepared["assistant_axis"] = unit(axis_vec)

    for r in ROLE_NAMES:
        raw = torch.load(
            f"{ROOT}/vectors/gemma-2-27b/role_vectors/{r}.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        direction = raw[TARGET_LAYER] - default_vec
        print(
            f"role {r}: raw_norm={raw[TARGET_LAYER].norm():.2f}  "
            f"(role-default)_norm={direction.norm():.2f}"
        )
        prepared[r] = unit(direction)

    # Multiple random unit vectors
    for i in range(N_RANDOM_VECTORS):
        torch.manual_seed(RANDOM_SEED_BASE + i)
        rand = torch.randn_like(axis_vec)
        prepared[f"random_{i}"] = unit(rand)
        print(f"random_{i}: seed={RANDOM_SEED_BASE + i}")

    # Save
    for name, v in prepared.items():
        torch.save(v, f"{ROOT}/vectors/steering/{name}_unit.pt")
        print(f"saved {name}_unit.pt  ||v||={v.norm():.6f}  dtype={v.dtype}")

    # Backward compat: also save random_0 as "random_unit.pt" so old scripts
    # that reference it still work (e.g. qual_check.py before migration)
    torch.save(prepared["random_0"], f"{ROOT}/vectors/steering/random_unit.pt")

    # Pairwise cosine similarity
    names = list(prepared.keys())
    n = len(names)
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            mat[i, j] = (prepared[names[i]] * prepared[names[j]]).sum()

    # Pretty-print (truncated for readability if many randoms)
    col_w = 14
    show_names = [n for n in names if not n.startswith("random_")] + ["random_0"]
    show_idxs = [names.index(n) for n in show_names]
    header = " " * col_w + "".join(f"{nm[:col_w-1]:>{col_w}}" for nm in show_names)
    print("\n=== Cosine similarity matrix (showing random_0 as representative) ===")
    print(header)
    for i in show_idxs:
        row = f"{names[i][:col_w-1]:<{col_w}}" + "".join(
            f"{mat[i, j].item():>{col_w}.4f}" for j in show_idxs
        )
        print(row)

    # Save full matrix
    out = {
        "names": names,
        "matrix": mat.tolist(),
        "target_layer": TARGET_LAYER,
    }
    with open(f"{ROOT}/results/vector_cosine_similarities.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nScript 1 done.")


if __name__ == "__main__":
    main()
