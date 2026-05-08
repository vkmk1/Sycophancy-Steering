"""Debug: verify steering actually perturbs activations, and check Gemma's A-bias."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ROOT, ASSISTANT_AXIS_PATH, MODEL_NAME, A_TOKEN_ID, B_TOKEN_ID, TARGET_LAYER

sys.path.insert(0, ASSISTANT_AXIS_PATH)
from assistant_axis.steering import ActivationSteering

A_ID, B_ID = A_TOKEN_ID, B_TOKEN_ID

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto')
model.eval()

with open(f'{ROOT}/data/eval_data.json') as f:
    eval_data = json.load(f)
eval_data = eval_data[:50]

# 1. Count how many have syc=A vs syc=B
a_syc = sum(1 for r in eval_data if r['sycophantic_answer'] == 'A')
print(f'Of 50 eval questions: syc=A in {a_syc}, syc=B in {50-a_syc}')

# 2. Baseline: what does the model pick?
choices_a = 0
for row in eval_data:
    prompt = tok.apply_chat_template([{'role':'user','content':row['question_text']}], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    lp = torch.log_softmax(out.logits[0,-1,:].float(), dim=-1)
    lp_a, lp_b = float(lp[A_ID]), float(lp[B_ID])
    if lp_a > lp_b:
        choices_a += 1
print(f'Baseline: model chose A {choices_a}/50 times')

# 3. Verify hook fires and perturbs activations.
axis_unit = torch.load(f'{ROOT}/vectors/steering/assistant_axis_unit.pt', weights_only=False).float().to(model.device)

captured = {}
def make_cap(tag):
    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured[tag] = t[0, -1, :].detach().float().clone()
        return None
    return hook

row = eval_data[0]
prompt = tok.apply_chat_template([{'role':'user','content':row['question_text']}], tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors='pt').to(model.device)

h1 = model.model.layers[TARGET_LAYER].register_forward_hook(make_cap('baseline'))
with torch.no_grad():
    out1 = model(**inputs)
h1.remove()

h2 = model.model.layers[TARGET_LAYER].register_forward_hook(make_cap('pre_steer'))
with ActivationSteering(model, steering_vectors=[axis_unit], coefficients=[5000.0], layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
    with torch.no_grad():
        out2 = model(**inputs)
h2.remove()

def make_post_cap():
    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured['post_steer'] = t[0, -1, :].detach().float().clone()
        return None
    return hook

with ActivationSteering(model, steering_vectors=[axis_unit], coefficients=[5000.0], layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
    h3 = model.model.layers[TARGET_LAYER].register_forward_hook(make_post_cap())
    with torch.no_grad():
        out3 = model(**inputs)
    h3.remove()

base = captured['baseline']
pre = captured['pre_steer']
post = captured['post_steer']
print(f'\nBaseline act norm: {base.norm():.2f}')
print(f'Pre-steer capture (inside ActSteer ctx): norm {pre.norm():.2f}')
print(f'Post-steer capture (inside ActSteer ctx): norm {post.norm():.2f}')
print(f'||pre - base|| = {(pre - base).norm():.4f}  (should be ~0)')
print(f'||post - base|| = {(post - base).norm():.4f}  (should be ~5000)')
print(f'||post - pre|| = {(post - pre).norm():.4f}  (should be ~5000 if steering applied)')

lp1 = torch.log_softmax(out1.logits[0,-1,:].float(), dim=-1)
lp2 = torch.log_softmax(out2.logits[0,-1,:].float(), dim=-1)
lp3 = torch.log_softmax(out3.logits[0,-1,:].float(), dim=-1)
print(f'\nBaseline q0: logp(A)={float(lp1[A_ID]):.3f}  logp(B)={float(lp1[B_ID]):.3f}')
print(f'Steer   q0: logp(A)={float(lp2[A_ID]):.3f}  logp(B)={float(lp2[B_ID]):.3f}')
print(f'Steer2  q0: logp(A)={float(lp3[A_ID]):.3f}  logp(B)={float(lp3[B_ID]):.3f}')

print('\n=== Free-generation comparison ===')
short_q = "Tell me a fun fact in 1 sentence."
prompt = tok.apply_chat_template([{'role':'user','content':short_q}], tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print('BASELINE:', tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

with ActivationSteering(model, steering_vectors=[axis_unit], coefficients=[5000.0], layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print('STEER +5000 (toward assistant):', tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

with ActivationSteering(model, steering_vectors=[axis_unit], coefficients=[-5000.0], layer_indices=[TARGET_LAYER], intervention_type='addition', positions='all'):
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print('STEER -5000 (toward role):', tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
