"""Extract every steering vector needed for the Qwen replication in one pass.

This combines what 02b_extract_caa.py + 01_prepare_steering_vectors.py do on
Gemma (which relied on a pre-computed `lu-christina/assistant-axis-vectors`
release that does not exist for Qwen 3 32B) and folds in a simplified
persona-contrast extraction for the 9 role vectors + assistant axis. Produces
everything in vectors/steering/ in the same file format the rest of the
pipeline expects.

Method for the persona/assistant vectors:
  For each role R in CRITICAL_ROLES + CONFORMIST_ROLES:
    1. Take the role's 5 system-prompt variants and 40 on-topic questions
       from assistant-axis/data/roles/instructions/{role}.json.
    2. For each (system_prompt, question) pair (= 200 pairs), run a single
       forward pass with the chat template applied and capture the
       last-prompt-token activation at TARGET_LAYER.
    3. role_mean_activation = mean over those 200 activations.
    4. Compute default_mean_activation the same way on the 240 extraction
       questions with NO system prompt (i.e. the default assistant persona).
    5. role_direction = role_mean - default_mean. Unit-normalise.

  assistant_axis_direction = default_mean - mean(all role_means).
  (Matches the sign convention of lu-christina/assistant-axis-vectors: the
  assistant axis points FROM the space of personas TOWARD the neutral
  assistant persona, so steering with a +coef pushes the residual "more
  assistant-like" and is expected to reduce sycophancy.)

Method for CAA: identical to 02b_extract_caa.py on Gemma (Rimsky et al. 2024;
contrast last-token activations on sycophantic vs honest answer appendings on
the NLP survey + political typology training set, 1000 pairs per dataset).

Method for random controls: 10 unit-Gaussian vectors seeded from
RANDOM_SEED_BASE + i (identical to 01_prepare_steering_vectors.py on Gemma).

Method for decomposition: once CAA + roles are all built, project each
role/axis vector onto unit-CAA, save the CAA-aligned component and the
CAA-orthogonal residual (both unit-normalised). This is exactly what
01_prepare_steering_vectors.py does on Gemma.

This script deliberately does NOT run the 1200-response generation +
OpenAI-judge step from assistant-axis/pipeline/; that pipeline is how
lu-christina/assistant-axis-vectors was built and it is not feasible to
re-run end-to-end in this session. Persona-contrast directions extracted
from mean hidden states are a well-studied, honest alternative (used e.g.
in Panickssery et al. / Rimsky et al. / Zou et al.); they may differ
quantitatively from the generation-derived vectors but the protocol is
applied identically across all 9 roles so the cross-condition comparison
remains internally valid.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ROOT, MODEL_NAME, TARGET_LAYER,
    CAA_DATASETS, CRITICAL_ROLES, CONFORMIST_ROLES, ROLE_NAMES,
    N_RANDOM_VECTORS, RANDOM_SEED_BASE,
)

ROLES_DATA_DIR = "/home/ubuntu/assistant-axis/data/roles/instructions"
EXTRACTION_QS_PATH = "/home/ubuntu/assistant-axis/data/extraction_questions.jsonl"
CAA_SUBSAMPLE_SEED = 2024
CAA_N_PER_DATASET = 1000


def unit(v):
    return v / (v.norm() + 1e-8)


def normalize_answer(ans):
    if isinstance(ans, list):
        ans = ans[0]
    return ans.strip().strip("()")


@torch.no_grad()
def run_and_capture(model, tokenizer, prompt, hook_state):
    """Run one forward pass; hook_state['act'] is filled by the layer hook."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model(**inputs)
    return hook_state["act"].clone()


def load_extraction_questions():
    qs = []
    with open(EXTRACTION_QS_PATH) as f:
        for line in f:
            qs.append(json.loads(line)["question"])
    return qs


def load_role_spec(role):
    path = f"{ROLES_DATA_DIR}/{role}.json"
    with open(path) as f:
        d = json.load(f)
    sys_prompts = [x["pos"] for x in d["instruction"]]
    qs = list(d["questions"])
    return sys_prompts, qs


def mean_activation_over(prompts, model, tokenizer, hook_state, label=""):
    acts = []
    for i, p in enumerate(prompts):
        if i % 50 == 0 and i > 0:
            print(f"    [{label}] {i}/{len(prompts)}")
        acts.append(run_and_capture(model, tokenizer, p, hook_state))
    return torch.stack(acts).mean(dim=0)


def build_chat(tokenizer, system_prompt, user_text):
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    # enable_thinking=False: Qwen 3's default chat template otherwise injects
    # a <think> block that distorts the last-token activation.
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def build_caa_pair_prompt(tokenizer, question, answer_letter):
    msgs = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt += f" ({answer_letter})"
    return prompt


def load_caa_pairs():
    pairs = []
    rng = random.Random(CAA_SUBSAMPLE_SEED)
    for fname in CAA_DATASETS:
        path = f"{ROOT}/data/sycophancy/{fname}"
        if not os.path.exists(path):
            print(f"WARN: {path} missing")
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
        file_pairs = file_pairs[:CAA_N_PER_DATASET]
        print(f"  CAA: {fname}: {len(file_pairs)} pairs")
        pairs.extend(file_pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-caa", action="store_true",
                    help="only build role + random vectors (CAA already exists)")
    ap.add_argument("--skip-personas", action="store_true",
                    help="only build CAA + random vectors (personas already exist)")
    args = ap.parse_args()

    os.makedirs(f"{ROOT}/vectors/steering", exist_ok=True)

    print(f"Loading model {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Model loaded.")

    hook_state = {"act": None}

    def cap_hook(module, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        hook_state["act"] = tensor[0, -1, :].detach().to(torch.float32).cpu()
        return None

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(cap_hook)

    # -------- Persona / assistant-axis extraction --------
    if not args.skip_personas:
        extraction_qs = load_extraction_questions()
        print(f"\nExtraction question bank: {len(extraction_qs)} qs")

        # Default persona = no system prompt, applied to the same extraction
        # question bank. Using a common question bank for role vs default keeps
        # the contrast direction aligned to persona style rather than topic.
        print("Computing default-assistant mean activation...")
        default_prompts = [build_chat(tokenizer, None, q) for q in extraction_qs]
        t0 = time.time()
        default_mean = mean_activation_over(
            default_prompts, model, tokenizer, hook_state, "default"
        )
        print(f"  default_mean ||.||={default_mean.norm():.3f}  "
              f"({(time.time()-t0)/60:.1f} min)")

        role_means = {}
        for r in ROLE_NAMES:
            print(f"\n--- Role: {r} ---")
            sys_prompts, qs = load_role_spec(r)
            # 5 sys prompts × 40 questions = 200 prompts per role.
            role_prompts = [
                build_chat(tokenizer, sp, q) for sp in sys_prompts for q in qs
            ]
            t0 = time.time()
            rm = mean_activation_over(
                role_prompts, model, tokenizer, hook_state, r
            )
            print(f"  role_mean_{r} ||.||={rm.norm():.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)")
            role_means[r] = rm

        # Assistant-axis direction: default_mean − mean(role_means).
        # Sign convention: +coef pushes activation toward the neutral
        # assistant-persona direction (same as Gemma's assistant_axis).
        roles_mean = torch.stack(list(role_means.values())).mean(dim=0)
        axis_direction = default_mean - roles_mean
        torch.save(
            unit(axis_direction),
            f"{ROOT}/vectors/steering/assistant_axis_unit.pt",
        )
        print(f"\nsaved assistant_axis_unit.pt  raw_norm={axis_direction.norm():.3f}")

        for r, rm in role_means.items():
            direction = rm - default_mean
            torch.save(
                unit(direction),
                f"{ROOT}/vectors/steering/{r}_unit.pt",
            )
            print(f"saved {r}_unit.pt  raw_norm={direction.norm():.3f}")

        # Save metadata on the simplified extraction for reviewers.
        meta = {
            "method": (
                "Persona-contrast extraction (simplified from "
                "assistant-axis pipeline). role_direction = "
                "mean(activation | persona_sys_prompt + extraction_q) - "
                "mean(activation | no_sys_prompt + extraction_q). "
                "Unit-normalised. Uses the 5 positive system prompts and "
                "40 role-specific questions from assistant-axis/data/roles/ "
                "(200 forward passes per role) and the 240 extraction_questions "
                "for the default."
            ),
            "target_layer": TARGET_LAYER,
            "model": MODEL_NAME,
            "assistant_axis_direction_sign": (
                "default_mean minus mean-of-role-means; +coef pushes toward "
                "the neutral-assistant persona."
            ),
            "roles": ROLE_NAMES,
        }
        with open(f"{ROOT}/vectors/steering/persona_extraction_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    # -------- CAA extraction --------
    if not args.skip_caa:
        pairs = load_caa_pairs()
        assert pairs, "No CAA pairs loaded"
        print(f"\nCAA: {len(pairs)} training pairs")

        syc_acts, hon_acts = [], []
        t0 = time.time()
        for i, p in enumerate(pairs):
            if i % 200 == 0:
                print(f"  CAA pair {i}/{len(pairs)}  "
                      f"({(time.time()-t0)/60:.1f} min)")
            syc_prompt = build_caa_pair_prompt(tokenizer, p["question"],
                                               p["sycophantic_answer"])
            hon_prompt = build_caa_pair_prompt(tokenizer, p["question"],
                                               p["honest_answer"])
            syc_acts.append(run_and_capture(model, tokenizer, syc_prompt, hook_state))
            hon_acts.append(run_and_capture(model, tokenizer, hon_prompt, hook_state))
        caa_raw = torch.stack(syc_acts).mean(dim=0) - torch.stack(hon_acts).mean(dim=0)
        torch.save(unit(caa_raw), f"{ROOT}/vectors/steering/caa_unit.pt")
        print(f"saved caa_unit.pt  raw_norm={caa_raw.norm():.3f}")
        with open(f"{ROOT}/vectors/steering/caa_metadata.json", "w") as f:
            json.dump({
                "method": "CAA (Rimsky et al. 2024)",
                "training_datasets": CAA_DATASETS,
                "n_pairs": len(pairs),
                "n_per_dataset_cap": CAA_N_PER_DATASET,
                "subsample_seed": CAA_SUBSAMPLE_SEED,
                "target_layer": TARGET_LAYER,
                "raw_norm": float(caa_raw.norm()),
                "direction": "FROM honest TOWARD sycophantic (steer with NEGATIVE coeff to reduce syc)",
            }, f, indent=2)

    handle.remove()

    # -------- Random controls --------
    # Same shape as the captured activation.
    hid = default_mean.numel() if not args.skip_personas else torch.load(
        f"{ROOT}/vectors/steering/caa_unit.pt", weights_only=False
    ).numel()
    for i in range(N_RANDOM_VECTORS):
        torch.manual_seed(RANDOM_SEED_BASE + i)
        rnd = torch.randn(hid).float()
        torch.save(unit(rnd), f"{ROOT}/vectors/steering/random_{i}_unit.pt")
    torch.save(
        torch.load(f"{ROOT}/vectors/steering/random_0_unit.pt", weights_only=False),
        f"{ROOT}/vectors/steering/random_unit.pt",
    )
    print(f"saved {N_RANDOM_VECTORS} random_*_unit.pt")

    # -------- Decomposition --------
    caa_unit = torch.load(f"{ROOT}/vectors/steering/caa_unit.pt", weights_only=False).float()
    print(f"\n=== CAA decomposition ===")
    decomp = {}
    for name in ["assistant_axis"] + ROLE_NAMES:
        v = torch.load(
            f"{ROOT}/vectors/steering/{name}_unit.pt", weights_only=False
        ).float()
        proj = float((v * caa_unit).sum())
        comp = proj * caa_unit
        resid = v - comp
        torch.save(unit(comp), f"{ROOT}/vectors/steering/{name}_caa_component_unit.pt")
        torch.save(unit(resid), f"{ROOT}/vectors/steering/{name}_residual_unit.pt")
        decomp[name] = {
            "cosine_with_caa": proj,
            "caa_component_norm": float(comp.norm()),
            "residual_norm": float(resid.norm()),
        }
        print(f"  {name:18s}  cos(CAA)={proj:+.4f}  "
              f"||comp||={comp.norm():.4f}  ||resid||={resid.norm():.4f}")
    with open(f"{ROOT}/vectors/steering/caa_decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2)

    # -------- Pairwise cosine similarity matrix --------
    names_all = ["assistant_axis"] + ROLE_NAMES + ["caa"] + [
        f"random_{i}" for i in range(N_RANDOM_VECTORS)
    ]
    vs = {}
    for n in names_all:
        vs[n] = torch.load(
            f"{ROOT}/vectors/steering/{n}_unit.pt", weights_only=False
        ).float()
    N = len(names_all)
    mat = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            mat[i, j] = (vs[names_all[i]] * vs[names_all[j]]).sum()
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/vector_cosine_similarities.json", "w") as f:
        json.dump({
            "names": names_all,
            "matrix": mat.tolist(),
            "target_layer": TARGET_LAYER,
        }, f, indent=2)

    print("\nExtraction done.")


if __name__ == "__main__":
    main()
