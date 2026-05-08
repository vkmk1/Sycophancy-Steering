"""Find a real eval-data row where skeptic +2000 AGREED with the user's
stated opinion (chose_sycophantic = True with positive syc_logit), then
re-run the same question as a free-form generation under skeptic +2000 to
show what the model actually says when it agrees."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, TARGET_LAYER

sys.path.insert(0, ASSISTANT_AXIS_PATH)
from assistant_axis.steering import ActivationSteering

with open(f'{ROOT}/results/checkpoints/skeptic.json') as f:
    rows = json.load(f)
with open(f'{ROOT}/data/eval_data.json') as f:
    eval_data = json.load(f)
qid_map = {r['question_id']: r for r in eval_data}

# Skeptic +2000 rows where the model chose the sycophantic answer with high confidence
agreed = [r for r in rows if r['coefficient'] == 2000.0 and r['chose_sycophantic']]
agreed.sort(key=lambda r: -r['syc_logit'])
print(f"Total skeptic+2000 rows where model agreed with user: {len(agreed)} / {sum(1 for r in rows if r['coefficient']==2000.0)}")

print("\nTop candidates (highest syc_logit while still agreeing):")
for i, r in enumerate(agreed[:5]):
    qrow = qid_map[r['question_id']]
    print(f"\n[{i}] qid={r['question_id']}  variant={qrow['variant']}  "
          f"syc_letter={r['sycophantic_answer']}  chose={r['chosen']}  "
          f"syc_logit={r['syc_logit']:+.2f}  logp(A)={r['logprob_A']:.2f}  logp(B)={r['logprob_B']:.2f}")
    print("USER:", qrow['question_text'][-500:])

if not agreed:
    print("No agreed rows found.")
    sys.exit(0)

pick = agreed[0]
qrow = qid_map[pick['question_id']]
print("\n" + "=" * 80)
print(f"Picked qid {pick['question_id']}.  Re-running as free-form generation under skeptic+2000.")
print("=" * 80)

q_text = qrow['question_text']
cuts = ['\n\nChoices:', '\n (A)', '\n(A)', '\nChoices:']
cut = len(q_text)
for c in cuts:
    j = q_text.find(c)
    if j > 0:
        cut = min(cut, j)
open_q = q_text[:cut].rstrip() + "\n\nDo you agree with my view? Briefly explain."
print("\nUSER PROMPT (free-form version):")
print(open_q)

print("\nLoading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()

skeptic = torch.load(f'{ROOT}/vectors/steering/skeptic_unit.pt', weights_only=False).float().to(model.device)

def gen(text, vec=None, coeff=0.0, max_new=180):
    chat = [{'role':'user','content':text}]
    p = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors='pt').to(model.device)
    if vec is None or coeff == 0.0:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    else:
        with ActivationSteering(model, steering_vectors=[vec], coefficients=[coeff],
                                layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n[BASELINE]")
print(gen(open_q))
print("\n[SKEPTIC +2000]")
print(gen(open_q, skeptic, 2000.0))
