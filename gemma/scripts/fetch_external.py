"""
One-shot external-dependency downloader.

Pulls the files that 00_setup.py, 02b_extract_caa.py, and 02_evaluate_steering.py
assume already exist on disk but do not download themselves:

  1. Role + axis vectors for Gemma 2 27B, from the HuggingFace model repo
     lu-christina/assistant-axis-vectors. These land in:
         vectors/gemma-2-27b/assistant_axis.pt
         vectors/gemma-2-27b/default_vector.pt
         vectors/gemma-2-27b/role_vectors/{role}.pt    for each role in config.ROLE_NAMES

  2. Sycophancy datasets from the anthropics/evals GitHub repo. The HuggingFace
     mirror Anthropic/model-written-evals has a duplicate-OID bug where
     philpapers2020 and nlp_survey share the same content, so we fetch
     authoritative jsonl from the GitHub blob store:
         data/sycophancy/sycophancy_on_philpapers2020.jsonl       (eval)
         data/sycophancy/sycophancy_on_nlp_survey.jsonl           (CAA training)
         data/sycophancy/sycophancy_on_political_typology_quiz.jsonl (CAA training)

Idempotent: skips files that already exist and validates file size > 0. Re-run
safely. Does NOT download the Gemma 2 27B weights -- those come via transformers
on first model-load in 00_setup.py / 02b_extract_caa.py / 02_evaluate_steering.py.

Requires:
  - HuggingFace auth for lu-christina/assistant-axis-vectors (public, but the
    huggingface_hub library expects HF_TOKEN or a cached token for some paths).
  - Internet access to api.github.com / raw.githubusercontent.com.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from config import ROOT, ROLE_NAMES, CAA_DATASETS, EVAL_DATASET


# --- Sycophancy datasets: authoritative GitHub blob shas (pin to immutable) ---
# These blob shas come from anthropics/evals as of the latest public commit on
# main. Pinning avoids a silent dataset change between runs.
ANTHROPIC_EVALS_BLOBS = {
    "sycophancy_on_philpapers2020.jsonl":      "5525210614d4f26b1732042e7dcb7210d23fe5aa",
    "sycophancy_on_nlp_survey.jsonl":          "480ff822079406be617c21979a3848fb29e5c0e2",
    "sycophancy_on_political_typology_quiz.jsonl": "52f57e31aece2d5126c49cc0e5a9a738b7513754",
}


def fetch_sycophancy_datasets(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    required = set([EVAL_DATASET]) | set(CAA_DATASETS)
    print(f"Required sycophancy files: {sorted(required)}")
    for fname, sha in ANTHROPIC_EVALS_BLOBS.items():
        if fname not in required:
            # skip datasets not used by this config (future-proof)
            continue
        out = os.path.join(out_dir, fname)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  [skip] {fname} already present ({os.path.getsize(out)} bytes)")
            continue
        url = f"https://api.github.com/repos/anthropics/evals/git/blobs/{sha}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github.raw"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            print(f"  [FAIL] {fname}: HTTP {e.code} from {url}")
            raise
        with open(out, "wb") as f:
            f.write(data)
        print(f"  [ok]   {fname} -> {out} ({len(data)} bytes)")


def fetch_role_vectors(out_dir):
    """Download the pre-computed gemma-2-27b vectors from the HF dataset
    lu-christina/assistant-axis-vectors (NOT a model repo -- pass
    repo_type='dataset' to hf_hub_download)."""
    from huggingface_hub import hf_hub_download  # lazy import

    repo_id = "lu-christina/assistant-axis-vectors"
    repo_type = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    role_dir = os.path.join(out_dir, "role_vectors")
    os.makedirs(role_dir, exist_ok=True)

    required = [("gemma-2-27b/assistant_axis.pt", os.path.join(out_dir, "assistant_axis.pt")),
                ("gemma-2-27b/default_vector.pt", os.path.join(out_dir, "default_vector.pt"))]
    for role in ROLE_NAMES:
        required.append((f"gemma-2-27b/role_vectors/{role}.pt",
                         os.path.join(role_dir, f"{role}.pt")))

    missing = []
    for hf_path, local_path in required:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            print(f"  [skip] {hf_path} already at {local_path}")
            continue
        try:
            src = hf_hub_download(repo_id=repo_id, filename=hf_path,
                                  repo_type=repo_type)
        except Exception as e:
            print(f"  [MISS] {hf_path}: {type(e).__name__} (will continue; "
                  "see summary)")
            missing.append(hf_path)
            continue
        with open(src, "rb") as r, open(local_path, "wb") as w:
            w.write(r.read())
        print(f"  [ok]   {hf_path} -> {local_path}")
    if missing:
        print(f"\n  NOTE: {len(missing)} file(s) not found in "
              f"{repo_type}:{repo_id}. These are printed above. The main "
              "pipeline will error on any condition whose vector is missing "
              "unless the vectors are provided by another means.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vectors", action="store_true",
                    help="skip HF role-vector download (use when only refreshing datasets)")
    ap.add_argument("--skip-datasets", action="store_true",
                    help="skip GitHub sycophancy-dataset download")
    args = ap.parse_args()

    if not args.skip_datasets:
        print("=== Sycophancy datasets (anthropics/evals) ===")
        fetch_sycophancy_datasets(f"{ROOT}/data/sycophancy")

    if not args.skip_vectors:
        print("\n=== Role + axis vectors (lu-christina/assistant-axis-vectors) ===")
        fetch_role_vectors(f"{ROOT}/vectors/gemma-2-27b")

    # Summary
    print("\n=== Summary ===")
    for fname in sorted(set([EVAL_DATASET]) | set(CAA_DATASETS)):
        p = f"{ROOT}/data/sycophancy/{fname}"
        ok = os.path.exists(p) and os.path.getsize(p) > 1000
        print(f"  {'OK ' if ok else 'MISS'} {p}")
    for vname in ["assistant_axis.pt", "default_vector.pt"]:
        p = f"{ROOT}/vectors/gemma-2-27b/{vname}"
        ok = os.path.exists(p) and os.path.getsize(p) > 1000
        print(f"  {'OK ' if ok else 'MISS'} {p}")
    for role in ROLE_NAMES:
        p = f"{ROOT}/vectors/gemma-2-27b/role_vectors/{role}.pt"
        ok = os.path.exists(p) and os.path.getsize(p) > 1000
        print(f"  {'OK ' if ok else 'MISS'} {p}")


if __name__ == "__main__":
    main()
