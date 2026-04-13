"""Rebuild eval_data.json with counterbalancing:
 - Only 2-choice (A/B) questions (drop 3-choice with a (C))
 - For each question, produce TWO variants: original + A/B swapped
 - Sycophantic label flips in the swapped variant
 - Result: 2*N rows, paired by base_id

This lets us cancel Gemma's positional (A) bias in the sycophancy measure.
"""
import json
import random
import re

from config import ROOT, EVAL_N, EVAL_SEED

N_BASE = EVAL_N
SEED = EVAL_SEED

with open(f"{ROOT}/data/sycophancy/sycophancy_on_philpapers2020.jsonl") as f:
    raw = [json.loads(line) for line in f]

def normalize(ans):
    if isinstance(ans, list):
        ans = ans[0]
    return ans.strip().strip("()")

def parse_ab(q):
    """Return (preamble, a_text, b_text, tail) for a 2-choice question, or None."""
    m = re.search(r"\n \(A\) (.*?)\n \(B\) (.*?)(\n \(C\)|\n\nAnswer:)", q, re.DOTALL)
    if not m:
        return None
    if m.group(3).startswith("\n (C)"):
        return None  # 3-choice question, skip
    a_text = m.group(1)
    b_text = m.group(2)
    preamble = q[: m.start()]
    tail = q[m.start(3):]
    return preamble, a_text, b_text, tail

def swap(preamble, a_text, b_text, tail):
    return f"{preamble}\n (A) {b_text}\n (B) {a_text}{tail}"

def original(preamble, a_text, b_text, tail):
    return f"{preamble}\n (A) {a_text}\n (B) {b_text}{tail}"

parsed = []
skipped = 0
for i, d in enumerate(raw):
    syc = normalize(d["answer_matching_behavior"])
    hon = normalize(d["answer_not_matching_behavior"])
    if syc not in {"A", "B"} or hon not in {"A", "B"} or syc == hon:
        skipped += 1
        continue
    parts = parse_ab(d["question"])
    if parts is None:
        skipped += 1
        continue
    parsed.append(
        {
            "orig_id": i,
            "parts": parts,
            "syc_letter": syc,  # A or B in original ordering
        }
    )

print(f"2-choice rows: {len(parsed)}  skipped: {skipped}")
rng = random.Random(SEED)
rng.shuffle(parsed)
parsed = parsed[:N_BASE]

eval_rows = []
for base_id, p in enumerate(parsed):
    orig_q = original(*p["parts"])
    swap_q = swap(*p["parts"])
    orig_syc = p["syc_letter"]
    swap_syc = "B" if orig_syc == "A" else "A"

    eval_rows.append(
        {
            "question_id": len(eval_rows),
            "base_id": base_id,
            "orig_id": p["orig_id"],
            "variant": "original",
            "question_text": orig_q,
            "sycophantic_answer": orig_syc,
            "non_sycophantic_answer": "B" if orig_syc == "A" else "A",
        }
    )
    eval_rows.append(
        {
            "question_id": len(eval_rows),
            "base_id": base_id,
            "orig_id": p["orig_id"],
            "variant": "swapped",
            "question_text": swap_q,
            "sycophantic_answer": swap_syc,
            "non_sycophantic_answer": "B" if swap_syc == "A" else "A",
        }
    )

print(f"Eval rows (paired): {len(eval_rows)}  ({N_BASE} base x 2 orderings)")
# Quick sanity: first base pair
print("\n--- base 0 original ---")
print(eval_rows[0]["question_text"][-400:])
print("syc:", eval_rows[0]["sycophantic_answer"])
print("\n--- base 0 swapped ---")
print(eval_rows[1]["question_text"][-400:])
print("syc:", eval_rows[1]["sycophantic_answer"])

a_syc = sum(1 for r in eval_rows if r["sycophantic_answer"] == "A")
print(f"\nsyc=A: {a_syc}  syc=B: {len(eval_rows)-a_syc}  (should be balanced)")

with open(f"{ROOT}/data/eval_data.json", "w") as f:
    json.dump(eval_rows, f, indent=2)
print(f"\nWrote {ROOT}/data/eval_data.json")
