"""Build Qwen 3 32B steering vectors from the OFFICIAL
lu-christina/assistant-axis-vectors/qwen-3-32b release.

This replaces the home-brewed persona-contrast extraction used in an earlier
iteration (kept at extract_all_vectors.py for reference). The official
release is identical in structure to the gemma-2-27b release that the Gemma
pipeline consumes:

  qwen-3-32b/assistant_axis.pt         [64, 5120] bfloat16
  qwen-3-32b/default_vector.pt         [64, 5120] bfloat16
  qwen-3-32b/role_vectors/{role}.pt    [64, 5120] bfloat16

So this script mirrors 01_prepare_steering_vectors.py on Gemma: for each
role, take layer TARGET_LAYER, subtract the default_vector at TARGET_LAYER,
unit-normalise, save. For assistant_axis, take layer TARGET_LAYER and
unit-normalise (no subtract — axis is already a direction).

CAA is extracted separately via extract_all_vectors.py --skip-personas (the
authors don't ship CAA; that's our own Rimsky-style extraction on top of
the sycophancy train set).

Random controls are 10 unit-Gaussian vectors at seed RANDOM_SEED_BASE + i.

Decomposition (projection onto CAA + residual) is computed and saved in the
same files the downstream pipeline expects.
"""
import json
import os
import sys

import torch

from config import (
    ROOT, TARGET_LAYER, ROLE_NAMES, CRITICAL_ROLES, CONFORMIST_ROLES,
    N_RANDOM_VECTORS, RANDOM_SEED_BASE,
)

# Official release cached by huggingface_hub:
SRC = ("/home/ubuntu/.cache/huggingface/hub/"
       "datasets--lu-christina--assistant-axis-vectors/snapshots/"
       "3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b")


def unit(v):
    return v / (v.norm() + 1e-8)


def main():
    os.makedirs(f"{ROOT}/vectors/steering", exist_ok=True)

    # ---- Load raw tensors (same structure as Gemma pipeline) ----
    axis = torch.load(f"{SRC}/assistant_axis.pt", map_location="cpu",
                      weights_only=False).float()
    default = torch.load(f"{SRC}/default_vector.pt", map_location="cpu",
                         weights_only=False).float()
    assert axis.shape[0] > TARGET_LAYER, f"axis only has {axis.shape[0]} layers"

    axis_vec = axis[TARGET_LAYER]
    default_vec = default[TARGET_LAYER]
    print(f"axis @ L{TARGET_LAYER}   shape={tuple(axis_vec.shape)}  norm={axis_vec.norm():.4f}")
    print(f"default @ L{TARGET_LAYER} norm={default_vec.norm():.4f}")

    prepared = {"assistant_axis": unit(axis_vec)}
    torch.save(prepared["assistant_axis"],
               f"{ROOT}/vectors/steering/assistant_axis_unit.pt")
    print(f"saved assistant_axis_unit.pt")

    # ---- Role vectors ----
    for r in ROLE_NAMES:
        raw = torch.load(f"{SRC}/role_vectors/{r}.pt", map_location="cpu",
                         weights_only=False).float()
        direction = raw[TARGET_LAYER] - default_vec
        role_type = "critical" if r in CRITICAL_ROLES else "conformist"
        print(f"role {r:14s} ({role_type}): "
              f"raw_norm={raw[TARGET_LAYER].norm():.2f}  "
              f"direction_norm={direction.norm():.2f}")
        prepared[r] = unit(direction)
        torch.save(prepared[r], f"{ROOT}/vectors/steering/{r}_unit.pt")

    # ---- Random controls ----
    for i in range(N_RANDOM_VECTORS):
        torch.manual_seed(RANDOM_SEED_BASE + i)
        rnd = torch.randn_like(axis_vec)
        prepared[f"random_{i}"] = unit(rnd)
        torch.save(prepared[f"random_{i}"],
                   f"{ROOT}/vectors/steering/random_{i}_unit.pt")
    torch.save(prepared["random_0"], f"{ROOT}/vectors/steering/random_unit.pt")

    # ---- CAA (must already be built by extract_all_vectors.py --skip-personas) ----
    caa_path = f"{ROOT}/vectors/steering/caa_unit.pt"
    if os.path.exists(caa_path):
        prepared["caa"] = torch.load(caa_path, map_location="cpu",
                                      weights_only=False).float()
        print(f"\nCAA loaded from {caa_path}  (layer noted in caa_metadata.json)")
    else:
        print("\nWARNING: caa_unit.pt not present -- run extract_all_vectors.py "
              "--skip-personas to extract CAA at TARGET_LAYER before decomposition.")
        return

    # ---- Decomposition ----
    print("\n=== CAA decomposition ===")
    caa_unit_vec = prepared["caa"]
    decomp = {}
    for name in ["assistant_axis"] + ROLE_NAMES:
        v = prepared[name]
        proj = float((v * caa_unit_vec).sum())
        comp = proj * caa_unit_vec
        resid = v - comp
        torch.save(unit(comp),
                   f"{ROOT}/vectors/steering/{name}_caa_component_unit.pt")
        torch.save(unit(resid),
                   f"{ROOT}/vectors/steering/{name}_residual_unit.pt")
        decomp[name] = {
            "cosine_with_caa": proj,
            "caa_component_norm": float(comp.norm()),
            "residual_norm": float(resid.norm()),
        }
        print(f"  {name:18s}  cos(CAA)={proj:+.4f}  "
              f"||comp||={comp.norm():.4f}  ||resid||={resid.norm():.4f}")
    with open(f"{ROOT}/vectors/steering/caa_decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2)

    # ---- Cosine similarity matrix ----
    names = list(prepared.keys())
    N = len(names)
    mat = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            mat[i, j] = (prepared[names[i]] * prepared[names[j]]).sum()
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/vector_cosine_similarities.json", "w") as f:
        json.dump({"names": names, "matrix": mat.tolist(),
                   "target_layer": TARGET_LAYER}, f, indent=2)

    # ---- Provenance note ----
    with open(f"{ROOT}/vectors/steering/persona_extraction_metadata.json", "w") as f:
        json.dump({
            "method": (
                "Official assistant_axis + default_vector + role_vectors taken "
                "from lu-christina/assistant-axis-vectors/qwen-3-32b at "
                "target_layer=32 (as configured in assistant_axis.models.MODEL_CONFIGS). "
                "role direction = role_vector[L] - default_vector[L], unit-normalised. "
                "assistant_axis = axis[L], unit-normalised."
            ),
            "source_hf_dataset": "lu-christina/assistant-axis-vectors",
            "source_subdir": "qwen-3-32b",
            "target_layer": TARGET_LAYER,
            "model": "Qwen/Qwen3-32B",
            "roles": ROLE_NAMES,
        }, f, indent=2)

    print("\nBuild from official vectors done.")


if __name__ == "__main__":
    main()
