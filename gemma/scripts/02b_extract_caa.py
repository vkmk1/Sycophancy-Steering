"""
Script 2b: Extract CAA (Contrastive Activation Addition) sycophancy vector.

Following Rimsky et al. (2024):
  1. Load NLP survey + political typology sycophancy A/B pairs (training set).
  2. For each question, run two forward passes:
     - Append the sycophantic answer token -> capture activation at target layer
     - Append the honest answer token -> capture activation at target layer
  3. CAA vector = mean(sycophantic activations) - mean(honest activations)
  4. This is the "pure sycophancy direction" — from honest toward sycophantic.
  5. Unit-normalize and save as caa_unit.pt.

The CAA vector serves as the TARGETED sycophancy baseline against which
general persona vectors are compared. It uses different datasets than the
eval set (philpapers2020), ensuring no data leakage.

NOTE: The CAA vector is steered with NEGATIVE coefficient to reduce sycophancy
(since it points FROM honest TOWARD sycophantic).
"""
import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER,
    A_TOKEN_ID, B_TOKEN_ID, CAA_DATASETS,
)

sys.path.insert(0, ASSISTANT_AXIS_PATH)


# Default subsampling cap per dataset. Rimsky et al. 2024 use ~500 A/B pairs
# for the sycophancy CAA direction; we default to 1000 per dataset (2000 total
# across NLP-survey + political-typology-quiz). The full union has ~20k pairs,
# which costs hours of extra GPU time for a mean direction whose per-pair
# marginal information is already low. Override via --n-per-dataset 0 to use
# the full union.
DEFAULT_N_PER_DATASET = 1000
CAA_SUBSAMPLE_SEED = 2024


def unit(v):
    return v / (v.norm() + 1e-8)


def normalize_answer(ans):
    if isinstance(ans, list):
        ans = ans[0]
    return ans.strip().strip("()")


def load_caa_data(n_per_dataset: int):
    """Load sycophancy A/B pairs from training datasets (NOT philpapers),
    subsampled to n_per_dataset per file (0 = no cap)."""
    all_pairs = []
    rng = random.Random(CAA_SUBSAMPLE_SEED)
    for fname in CAA_DATASETS:
        path = f"{ROOT}/data/sycophancy/{fname}"
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        file_pairs = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                syc = normalize_answer(d["answer_matching_behavior"])
                hon = normalize_answer(d["answer_not_matching_behavior"])
                if syc not in {"A", "B"} or hon not in {"A", "B"} or syc == hon:
                    continue
                file_pairs.append({
                    "question": d["question"],
                    "sycophantic_answer": syc,
                    "honest_answer": hon,
                })
        rng.shuffle(file_pairs)
        if n_per_dataset and n_per_dataset > 0:
            file_pairs = file_pairs[:n_per_dataset]
        print(f"  {fname}: {len(file_pairs)} pairs (subsample cap={n_per_dataset})")
        all_pairs.extend(file_pairs)
    print(f"Loaded {len(all_pairs)} CAA training pairs from {len(CAA_DATASETS)} datasets")
    return all_pairs


def build_prompt(tokenizer, question_text, answer_letter):
    """Build prompt with the answer token appended (for activation capture)."""
    chat = [{"role": "user", "content": question_text}]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # Append the answer token so the model processes it
    prompt += f" ({answer_letter})"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-dataset", type=int, default=DEFAULT_N_PER_DATASET,
                    help=f"Cap on pairs per training file. "
                         f"0 = use all pairs (default: {DEFAULT_N_PER_DATASET}).")
    args = ap.parse_args()
    pairs = load_caa_data(args.n_per_dataset)
    if not pairs:
        print("ERROR: No CAA training data found. Download the sycophancy datasets first.")
        print("  Expected files in data/sycophancy/:")
        for f in CAA_DATASETS:
            print(f"    {f}")
        sys.exit(1)

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Register a hook to capture activations at the target layer
    captured = {}

    def capture_hook(module, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        # Capture the last token position activation
        captured["act"] = tensor[0, -1, :].detach().to(torch.float32).cpu()
        return None

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(capture_hook)

    # Collect activations for sycophantic and honest answers
    syc_activations = []
    hon_activations = []

    print(f"Extracting activations from {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}...")

        # Forward pass with sycophantic answer appended
        prompt_syc = build_prompt(tokenizer, pair["question"], pair["sycophantic_answer"])
        inputs_syc = tokenizer(prompt_syc, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs_syc)
        syc_activations.append(captured["act"].clone())

        # Forward pass with honest answer appended
        prompt_hon = build_prompt(tokenizer, pair["question"], pair["honest_answer"])
        inputs_hon = tokenizer(prompt_hon, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs_hon)
        hon_activations.append(captured["act"].clone())

    handle.remove()

    # CAA vector = mean(sycophantic) - mean(honest)
    # This points FROM honest TOWARD sycophantic
    syc_mean = torch.stack(syc_activations).mean(dim=0)
    hon_mean = torch.stack(hon_activations).mean(dim=0)
    caa_raw = syc_mean - hon_mean
    caa_unit_vec = unit(caa_raw)

    print(f"\nCAA vector computed:")
    print(f"  Shape: {tuple(caa_unit_vec.shape)}")
    print(f"  Raw norm: {caa_raw.norm():.4f}")
    print(f"  Unit norm: {caa_unit_vec.norm():.6f}")
    print(f"  N sycophantic activations: {len(syc_activations)}")
    print(f"  N honest activations: {len(hon_activations)}")

    # Save
    os.makedirs(f"{ROOT}/vectors/steering", exist_ok=True)
    torch.save(caa_unit_vec, f"{ROOT}/vectors/steering/caa_unit.pt")
    print(f"  Saved to vectors/steering/caa_unit.pt")

    # Compute cosine similarity with assistant axis for reference
    axis_path = f"{ROOT}/vectors/steering/assistant_axis_unit.pt"
    if os.path.exists(axis_path):
        axis = torch.load(axis_path, weights_only=False).float()
        cos_sim = float((caa_unit_vec * axis).sum())
        print(f"\n  Cosine similarity(CAA, assistant_axis) = {cos_sim:+.4f}")
        print(f"  (Negative means CAA points AWAY from the assistant direction,")
        print(f"   which makes sense: sycophancy = away from grounded assistant)")

    # Save metadata
    meta = {
        "method": "CAA (Rimsky et al. 2024)",
        "training_datasets": CAA_DATASETS,
        "n_per_dataset_cap": args.n_per_dataset,
        "n_pairs": len(pairs),
        "subsample_seed": CAA_SUBSAMPLE_SEED,
        "target_layer": TARGET_LAYER,
        "raw_norm": float(caa_raw.norm()),
        "direction": "FROM honest TOWARD sycophantic (steer with NEGATIVE coeff to reduce syc)",
    }
    with open(f"{ROOT}/vectors/steering/caa_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nScript 2b done.")


if __name__ == "__main__":
    main()
