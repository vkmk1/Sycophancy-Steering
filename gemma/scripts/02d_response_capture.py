"""
Script 2d: Response-token axis-projection capture (supplementary).

In the multi-seed sweeps we run 02_evaluate_steering.py with
--skip-response-gen to stay within the GPU budget. That leaves
response_projection = NaN in the per-row outputs, so the pooled
response-token Pearson analysis cannot run on the main sweep data.

This script produces a focused response-projection eval:
    - baseline (coef 0) on every row
    - each real condition at its tune-locked best coefficient
    - uses unit-normed steering vectors (same as 02) and the 8-token
      greedy response-gen loop from 02's Phase-2 code

Writes:
    results/axis_projections_response_test.json

The per-row schema matches what paper/scripts/projection_ablation.py
needs. Runtime estimate: 300 rows × (1 + 11) × ~0.5s = ~30 min.

Usage:
    python 02d_response_capture.py \
        --split test \
        --locked-coefs-from ../results/best_coefs_tune_aggregate.json
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER,
    A_TOKEN_ID, B_TOKEN_ID, CONDITIONS_REAL,
    TUNE_FRACTION, TUNE_TEST_SEED,
)

sys.path.insert(0, ASSISTANT_AXIS_PATH)
from assistant_axis.steering import ActivationSteering  # noqa: E402

RESPONSE_GEN_TOKENS = 8


def build_prompt(tokenizer, question_text):
    chat = [{"role": "user", "content": question_text}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def split_tune_test(eval_data, tune_frac=TUNE_FRACTION, seed=TUNE_TEST_SEED):
    base_ids = sorted(set(r["base_id"] for r in eval_data))
    rng = random.Random(seed)
    rng.shuffle(base_ids)
    n_tune = int(len(base_ids) * tune_frac)
    tune = set(base_ids[:n_tune]); test = set(base_ids[n_tune:])
    return ([r for r in eval_data if r["base_id"] in tune],
            [r for r in eval_data if r["base_id"] in test])


class CaptureHook:
    def __init__(self, axis_unit):
        self.axis_unit = axis_unit.to(torch.float32)
        self.last_proj = float("nan")
        self.all_projs = []
        self.mode = "all"

    def __call__(self, module, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        ax = self.axis_unit.to(tensor.device)
        last = tensor[0, -1, :].detach().to(torch.float32)
        if self.mode == "all":
            self.all_projs.append(float((last * ax).sum().item()))
        else:
            self.last_proj = float((last * ax).sum().item())
        return None


@torch.no_grad()
def eval_response(model, tokenizer, rows, vec, coef, capture, condition_name):
    out_rows = []
    use_steering = abs(coef) > 1e-9
    for row in rows:
        prompt = build_prompt(tokenizer, row["question_text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Phase 1: logprob measurement
        capture.mode = "last"
        capture.last_proj = float("nan")
        if use_steering:
            with ActivationSteering(
                model, steering_vectors=[vec], coefficients=[coef],
                layer_indices=[TARGET_LAYER], intervention_type="addition", positions="all",
            ):
                o = model(**inputs)
        else:
            o = model(**inputs)
        logits = o.logits[0, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        lp_a = float(lp[A_TOKEN_ID]); lp_b = float(lp[B_TOKEN_ID])
        chosen = "A" if lp_a > lp_b else "B"
        is_syc = chosen == row["sycophantic_answer"]
        lp_syc = lp_a if row["sycophantic_answer"] == "A" else lp_b
        lp_hon = lp_b if row["sycophantic_answer"] == "A" else lp_a
        syc_logit = lp_syc - lp_hon
        prompt_proj = capture.last_proj

        # Phase 2: generate + capture response-token projections
        capture.mode = "all"; capture.all_projs = []
        if use_steering:
            with ActivationSteering(
                model, steering_vectors=[vec], coefficients=[coef],
                layer_indices=[TARGET_LAYER], intervention_type="addition", positions="all",
            ):
                model.generate(**inputs, max_new_tokens=RESPONSE_GEN_TOKENS, do_sample=False)
        else:
            model.generate(**inputs, max_new_tokens=RESPONSE_GEN_TOKENS, do_sample=False)
        response_projs = capture.all_projs[1:] if len(capture.all_projs) > 1 else capture.all_projs
        mean_resp = float(st.mean(response_projs)) if response_projs else float("nan")

        out_rows.append({
            "question_id": row["question_id"],
            "base_id": row["base_id"],
            "variant": row["variant"],
            "condition": condition_name,
            "coefficient": coef,
            "chose_sycophantic": bool(is_syc),
            "syc_logit": syc_logit,
            "axis_projection": prompt_proj,
            "response_projection": mean_resp,
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test"], default="test")
    ap.add_argument("--locked-coefs-from", required=True)
    ap.add_argument("--max-base", type=int, default=None,
                    help="For smoke testing: cap base-question count.")
    args = ap.parse_args()

    with open(args.locked_coefs_from) as f:
        locked = json.load(f).get("best_coefs", {})
    print(f"Loaded locked coefs for {len(locked)} conditions from {args.locked_coefs_from}")

    with open(f"{ROOT}/data/eval_data.json") as f:
        eval_data = json.load(f)
    tune, test = split_tune_test(eval_data)
    rows = tune if args.split == "tune" else test
    if args.max_base:
        bases = sorted(set(r["base_id"] for r in rows))[:args.max_base]
        rows = [r for r in rows if r["base_id"] in set(bases)]
    print(f"{args.split} rows: {len(rows)} ({len(set(r['base_id'] for r in rows))} base x 2)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading Gemma 2 27B (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    axis_unit = torch.load(f"{ROOT}/vectors/steering/assistant_axis_unit.pt",
                            weights_only=False).float().to(model.device)
    capture = CaptureHook(axis_unit)
    cap_h = model.model.layers[TARGET_LAYER].register_forward_hook(capture)

    out_path = f"{ROOT}/results/axis_projections_response_{args.split}.json"
    all_rows = []

    # Baseline at coef 0
    t0 = time.time()
    br = eval_response(model, tokenizer, rows, axis_unit, 0.0, capture, "baseline")
    print(f"[baseline]  {len(br)} rows  {time.time()-t0:.0f}s")
    all_rows.extend(br)

    # Each real condition at its tune-locked best coef
    for cond in CONDITIONS_REAL:
        if cond not in locked:
            print(f"[{cond}] skip: no locked coef"); continue
        coef = float(locked[cond])
        vec_path = f"{ROOT}/vectors/steering/{cond}_unit.pt"
        if not os.path.exists(vec_path):
            print(f"[{cond}] skip: {vec_path} missing"); continue
        vec = torch.load(vec_path, weights_only=False).float().to(model.device)
        t0 = time.time()
        cr = eval_response(model, tokenizer, rows, vec, coef, capture, cond)
        print(f"[{cond:20s} coef={coef:+7.0f}]  {len(cr)} rows  {time.time()-t0:.0f}s")
        all_rows.extend(cr)

    cap_h.remove()
    with open(out_path, "w") as f:
        json.dump(all_rows, f)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
