"""
Script 1: Prepare steering vectors.

- Extract axis @ target layer (already a direction: default - role_mean)
- For each role (critical + conformist): direction = role - default, unit-normalize
- Generate N_RANDOM_VECTORS random unit control vectors
- If CAA vector exists (from 02b), include it in the cosine similarity matrix
- Compute and save decomposition: for each role vector, project onto CAA
  to get the CAA-aligned component and the residual
- Write all normalized vectors + pairwise cosine similarity matrix
"""
import json
import os

import torch

from config import (
    ROOT, TARGET_LAYER, ROLE_NAMES, CRITICAL_ROLES, CONFORMIST_ROLES,
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
        role_type = "critical" if r in CRITICAL_ROLES else "conformist"
        print(
            f"role {r} ({role_type}): raw_norm={raw[TARGET_LAYER].norm():.2f}  "
            f"(role-default)_norm={direction.norm():.2f}"
        )
        prepared[r] = unit(direction)

    # Multiple random unit vectors
    for i in range(N_RANDOM_VECTORS):
        torch.manual_seed(RANDOM_SEED_BASE + i)
        rand = torch.randn_like(axis_vec)
        prepared[f"random_{i}"] = unit(rand)

    # Check if CAA vector exists (produced by 02b_extract_caa.py)
    caa_path = f"{ROOT}/vectors/steering/caa_unit.pt"
    if os.path.exists(caa_path):
        caa = torch.load(caa_path, map_location="cpu", weights_only=False).float()
        prepared["caa"] = caa
        print(f"CAA vector loaded from {caa_path}")
    else:
        print(f"CAA vector not found at {caa_path} -- run 02b_extract_caa.py first")
        print("  (continuing without CAA)")

    # Save all vectors
    for name, v in prepared.items():
        torch.save(v, f"{ROOT}/vectors/steering/{name}_unit.pt")
        print(f"saved {name}_unit.pt  ||v||={v.norm():.6f}  dtype={v.dtype}")

    # Backward compat
    if "random_0" in prepared:
        torch.save(prepared["random_0"], f"{ROOT}/vectors/steering/random_unit.pt")

    # ---- Decomposition: project each role onto CAA ----
    if "caa" in prepared:
        print("\n=== CAA Decomposition ===")
        caa_unit_vec = prepared["caa"]
        decomp = {}
        for name in ["assistant_axis"] + ROLE_NAMES:
            v = prepared[name]
            # Component along CAA direction
            proj_scalar = float((v * caa_unit_vec).sum())
            caa_component = proj_scalar * caa_unit_vec
            residual = v - caa_component
            # Save components
            torch.save(unit(caa_component), f"{ROOT}/vectors/steering/{name}_caa_component_unit.pt")
            torch.save(unit(residual), f"{ROOT}/vectors/steering/{name}_residual_unit.pt")
            decomp[name] = {
                "cosine_with_caa": proj_scalar,
                "caa_component_norm": float(caa_component.norm()),
                "residual_norm": float(residual.norm()),
            }
            print(f"  {name}: cos(CAA)={proj_scalar:+.4f}  "
                  f"||caa_comp||={caa_component.norm():.4f}  "
                  f"||residual||={residual.norm():.4f}")
        with open(f"{ROOT}/vectors/steering/caa_decomposition.json", "w") as f:
            json.dump(decomp, f, indent=2)

    # ---- Pairwise cosine similarity ----
    names = list(prepared.keys())
    n = len(names)
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            mat[i, j] = (prepared[names[i]] * prepared[names[j]]).sum()

    # Pretty-print (show representative subset)
    col_w = 14
    show_names = [n for n in names if not n.startswith("random_")]
    if "random_0" in names:
        show_names.append("random_0")
    show_idxs = [names.index(n) for n in show_names if n in names]
    header = " " * col_w + "".join(f"{nm[:col_w-1]:>{col_w}}" for nm in show_names)
    print(f"\n=== Cosine similarity matrix ===")
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
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/vector_cosine_similarities.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nScript 1 done.")


if __name__ == "__main__":
    main()
