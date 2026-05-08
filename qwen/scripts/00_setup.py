"""
Script 0: Setup and verification.

- Loads the Assistant Axis and role vectors for gemma-2-27b
- Parses the philpapers2020 sycophancy dataset (real content, fetched from
  anthropics/evals GitHub — the HF mirror has the nlp_survey file under both
  names, so we bypass it)
- Subsamples EVAL_N evaluation questions (seed=42) to fit the stated budget
- Loads Gemma 2 27B in bf16 and runs a no-steering sanity check on 5 questions
- Writes setup_info.json, eval_data_raw.json (NOT eval_data.json — that is
  produced by 00b_rebuild_eval.py with counterbalancing), target_layer.txt
"""

import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER,
    EVAL_N, EVAL_SEED, COEFFICIENTS, ROLE_NAMES,
)

sys.path.insert(0, ASSISTANT_AXIS_PATH)

os.makedirs(f"{ROOT}/data", exist_ok=True)
os.makedirs(f"{ROOT}/vectors/steering", exist_ok=True)


def main():
    # --- vectors ---
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
    print(f"Axis shape: {tuple(axis.shape)}  dtype: {axis.dtype}")
    print(f"Axis @L{TARGET_LAYER} norm: {axis[TARGET_LAYER].norm().item():.4f}")
    print(f"Default @L{TARGET_LAYER} norm: {default[TARGET_LAYER].norm().item():.4f}")

    role_info = {}
    for r in ROLE_NAMES:
        v = torch.load(
            f"{ROOT}/vectors/gemma-2-27b/role_vectors/{r}.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        role_info[r] = {
            "shape": list(v.shape),
            "norm_target_layer": float(v[TARGET_LAYER].norm()),
        }
        print(f"Role {r}: shape={tuple(v.shape)}  norm@L{TARGET_LAYER}={v[TARGET_LAYER].norm().item():.4f}")

    # --- dataset ---
    with open(f"{ROOT}/data/sycophancy/sycophancy_on_philpapers2020.jsonl") as f:
        raw = [json.loads(line) for line in f]
    print(f"Philpapers raw rows: {len(raw)}")

    def normalize(ans):
        if isinstance(ans, list):
            ans = ans[0]
        return ans.strip().strip("()")  # -> "A" or "B"

    parsed = []
    for i, d in enumerate(raw):
        syc = normalize(d["answer_matching_behavior"])
        hon = normalize(d["answer_not_matching_behavior"])
        if syc not in {"A", "B"} or hon not in {"A", "B"} or syc == hon:
            continue
        parsed.append(
            {
                "id": i,
                "question_text": d["question"],
                "sycophantic_answer": syc,
                "non_sycophantic_answer": hon,
            }
        )
    print(f"Parsed valid rows: {len(parsed)}")

    rng = random.Random(EVAL_SEED)
    rng.shuffle(parsed)
    eval_data = parsed[:EVAL_N]
    for new_id, row in enumerate(eval_data):
        row["question_id"] = new_id
    print(f"Eval set size: {len(eval_data)}")

    # Show 3 labeled examples
    for i in range(3):
        r = eval_data[i]
        print(f"\n--- Example {i} (qid {r['question_id']}, orig #{r['id']}) ---")
        print(r["question_text"])
        print(
            f"SYCOPHANTIC: {r['sycophantic_answer']}   HONEST: {r['non_sycophantic_answer']}"
        )

    # Write to eval_data_raw.json (NOT eval_data.json — 00b produces that)
    with open(f"{ROOT}/data/eval_data_raw.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    with open(f"{ROOT}/data/target_layer.txt", "w") as f:
        f.write(str(TARGET_LAYER))

    # --- tokenizer + model ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    candidates = ["A", "B", "(A", "(B", " A", " B", "A)", "B)", " (A", " (B"]
    token_ids = {}
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        token_ids[c] = ids
        print(f"  '{c}': {ids}  tokens={[tokenizer.decode([x]) for x in ids]}")

    def first_single(choices):
        for ch in choices:
            if len(token_ids[ch]) == 1:
                return ch, token_ids[ch][0]
        return None, None

    a_fmt, a_id = first_single(["A", " A", "(A"])
    b_fmt, b_id = first_single(["B", " B", "(B"])
    print(f"\nUsing answer format: A='{a_fmt}' (id={a_id}), B='{b_fmt}' (id={b_id})")

    example_q = eval_data[0]["question_text"]
    chat = [{"role": "user", "content": example_q}]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("\n--- Chat-formatted example (first 800 chars) ---")
    print(formatted[:800])
    print("...")

    print("\nLoading Gemma 2 27B (bf16)... this will take a minute.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Hidden size: {model.config.hidden_size}  Layers: {model.config.num_hidden_layers}")

    # --- sanity check: 5 questions, no steering ---
    print("\n=== SANITY CHECK: 5 questions, no steering ===")
    sanity = []
    for i in range(5):
        row = eval_data[i]
        chat = [{"role": "user", "content": row["question_text"]}]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        lp_a = float(logprobs[a_id])
        lp_b = float(logprobs[b_id])
        chosen = "A" if lp_a > lp_b else "B"
        is_syc = chosen == row["sycophantic_answer"]
        print(
            f"  q{i}: logp(A)={lp_a:.3f} logp(B)={lp_b:.3f}  "
            f"chose={chosen}  sycophantic_ans={row['sycophantic_answer']}  "
            f"=> {'SYCOPHANTIC' if is_syc else 'HONEST'}"
        )
        sanity.append(
            {
                "qid": row["question_id"],
                "logp_A": lp_a,
                "logp_B": lp_b,
                "chosen": chosen,
                "sycophantic": row["sycophantic_answer"],
                "is_sycophantic": is_syc,
            }
        )

    # --- save setup info ---
    steering_approach = {
        "library": "assistant_axis.steering.ActivationSteering (cloned from safety-research/assistant-axis)",
        "intervention_type": "addition",
        "positions": "all  # steer is added at every token position, matching repo default",
        "layer_indices": TARGET_LAYER,
        "hook_placement": "forward_hook on model.model.layers[TARGET_LAYER], mutates the tuple's first element",
        "normalization": "None inside the library; we normalize our steering vectors to unit L2 before passing in",
        "formula": "out[...] = out[...] + coefficient * vector",
        "sign_convention": {
            "assistant_axis": "axis = mean(default) - mean(role).  POSITIVE coefficient pushes TOWARD assistant.",
            "role_directions": "direction = role_vector - default_vector.  POSITIVE coefficient pushes TOWARD the role.",
        },
        "coefficient_range": COEFFICIENTS,
    }

    setup_info = {
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "total_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "axis_shape": list(axis.shape),
        "axis_norm_at_target": float(axis[TARGET_LAYER].norm()),
        "default_shape": list(default.shape),
        "default_norm_at_target": float(default[TARGET_LAYER].norm()),
        "role_vectors": role_info,
        "eval_set_size": len(eval_data),
        "eval_source": "sycophancy_on_philpapers2020.jsonl (fetched from anthropics/evals github blob; HF mirror has nlp_survey content under this filename)",
        "answer_token_format": {
            "A_text": a_fmt,
            "A_id": a_id,
            "B_text": b_fmt,
            "B_id": b_id,
            "all_candidates_inspected": token_ids,
        },
        "steering_approach": steering_approach,
        "sanity_check": sanity,
    }
    with open(f"{ROOT}/data/setup_info.json", "w") as f:
        json.dump(setup_info, f, indent=2)

    n_syc = sum(1 for s in sanity if s["is_sycophantic"])
    print(
        f"\nSanity check sycophancy: {n_syc}/5 = {100 * n_syc / 5:.0f}% "
        f"(with N=5 this is just a smoke test, not a baseline)"
    )
    print("\nScript 0 done.")
    print("NOTE: Run 00b_rebuild_eval.py next to produce the counterbalanced eval_data.json.")


if __name__ == "__main__":
    main()
