"""Qualitative check: at the +2000 operating point, do skeptic-steered
generations actually disagree with the user (real sycophancy reduction), or
are they just noise (degradation)?

Compare generations from baseline, skeptic+2000, and random+2000 on the same
sycophancy prompts."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER

sys.path.insert(0, ASSISTANT_AXIS_PATH)
from assistant_axis.steering import ActivationSteering

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()

with open(f'{ROOT}/data/eval_data.json') as f:
    eval_data = json.load(f)

# Pick a few questions with strong baseline sycophancy
with open(f'{ROOT}/results/checkpoints/baseline.json') as f:
    baseline = json.load(f)
candidates = sorted(
    [r for r in baseline if r['variant'] == 'original' and r['chose_sycophantic']],
    key=lambda r: -r['syc_logit'],
)[:5]

skeptic = torch.load(f'{ROOT}/vectors/steering/skeptic_unit.pt', weights_only=False).float().to(model.device)
rand = torch.load(f'{ROOT}/vectors/steering/random_0_unit.pt', weights_only=False).float().to(model.device)
axis = torch.load(f'{ROOT}/vectors/steering/assistant_axis_unit.pt', weights_only=False).float().to(model.device)
da = torch.load(f'{ROOT}/vectors/steering/devils_advocate_unit.pt', weights_only=False).float().to(model.device)

def gen(prompt_text, vec=None, coeff=0.0, max_new=120):
    chat = [{'role': 'user', 'content': prompt_text}]
    p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(p, return_tensors='pt').to(model.device)
    if vec is None or coeff == 0.0:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    else:
        with ActivationSteering(model, steering_vectors=[vec], coefficients=[coeff],
                                layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def open_form(question_text):
    cuts = ['\n\nChoices:', '\n (A)', '\nChoices:', '\n(A)']
    cut = len(question_text)
    for c in cuts:
        i = question_text.find(c)
        if i > 0:
            cut = min(cut, i)
    prefix = question_text[:cut].rstrip()
    return prefix + "\n\nDo you agree with my view? Briefly explain."

for qi, row in enumerate(candidates):
    qrow = next(r for r in eval_data if r['question_id'] == row['question_id'])
    open_q = open_form(qrow['question_text'])
    print(f"\n========== QUESTION {qi+1} (qid {row['question_id']}, baseline syc_logit={row['syc_logit']:+.2f}) ==========")
    print("USER:", open_q[:600], '...' if len(open_q) > 600 else '')

    print("\n--- BASELINE ---")
    print(gen(open_q))
    print("\n--- SKEPTIC +2000 ---")
    print(gen(open_q, skeptic, 2000.0))
    print("\n--- DEVILS_ADVOCATE +2000 ---")
    print(gen(open_q, da, 2000.0))
    print("\n--- RANDOM +2000 (control) ---")
    print(gen(open_q, rand, 2000.0))
    print("\n--- ASSISTANT_AXIS +2000 ---")
    print(gen(open_q, axis, 2000.0))
