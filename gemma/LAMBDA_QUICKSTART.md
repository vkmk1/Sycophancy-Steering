# Lambda Quickstart -- Copy-Paste Commands

**Time**: ~10 hours total. **Cost**: ~$25 on H100.

Everything below is meant to be copy-pasted in order. Do NOT skip steps.
Do NOT run steps in parallel. Always work inside tmux.

---

## 1. First-time setup (do once)

```bash
# SSH in
ssh ubuntu@<YOUR_LAMBDA_IP>

# Start tmux (CRITICAL -- protects against SSH drops)
tmux new-session -s exp

# Clone repos
cd /home/ubuntu
git clone <YOUR_REPO_URL> experiment
git clone https://github.com/safety-research/assistant-axis.git assistant-axis

# Install deps
pip install transformers accelerate huggingface_hub datasets scipy scikit-learn statsmodels matplotlib seaborn

# HuggingFace auth (paste your token when asked)
huggingface-cli login

# Set env vars
echo 'export EXPERIMENT_ROOT=/home/ubuntu/experiment' >> ~/.bashrc
echo 'export ASSISTANT_AXIS_PATH=/home/ubuntu/assistant-axis' >> ~/.bashrc
source ~/.bashrc

# Go to scripts dir (ALL commands below run from here)
cd /home/ubuntu/experiment/scripts
```

## 2. Data setup (~15 min)

```bash
python 00_setup.py

curl -sLk "https://api.github.com/repos/anthropics/evals/git/blobs/5525210614d4f26b1732042e7dcb7210d23fe5aa" \
     -H "Accept: application/vnd.github.raw" \
     -o ../data/sycophancy/sycophancy_on_philpapers2020.jsonl

python 00b_rebuild_eval.py
```

## 3. Vectors (~30 min)

```bash
python 02b_extract_caa.py
python 01_prepare_steering_vectors.py
```

## 4. Smoke test (~5 min)

```bash
python 02_evaluate_steering.py --max-base 5
```

If this succeeds, the full run will work. If it fails, fix the error before continuing.

## 5. Full evaluation (~6-8 hours)

```bash
python 02_evaluate_steering.py 2>&1 | tee ../results/eval_log.txt
```

If interrupted, resume with:
```bash
python 02_evaluate_steering.py --resume 2>&1 | tee -a ../results/eval_log.txt
```

## 6. Analysis (~5 min)

```bash
python 03_analysis.py
cat ../results/summary.txt
```

## 7. Qualitative (~45 min)

```bash
python over_correction_check.py 2>&1 | tee ../results/overcorrection_log.txt
python qual_check.py 2>&1 | tee ../results/qualcheck_log.txt
```

## 8. Download results (from your local machine)

```bash
scp -r ubuntu@<LAMBDA_IP>:/home/ubuntu/experiment/results ./results/
scp -r ubuntu@<LAMBDA_IP>:/home/ubuntu/experiment/figures ./figures/
```

## 9. STOP THE INSTANCE

Go to https://cloud.lambdalabs.com/instances and terminate. You're paying by the hour.

---

## If SSH disconnects

```bash
ssh ubuntu@<LAMBDA_IP>
tmux attach -t exp
# Your run is still going
```

## If a step fails

| Error | Fix |
|-------|-----|
| CUDA out of memory | Run `nvidia-smi`, kill other processes, retry |
| FileNotFoundError: eval_data.json | Run `python 00b_rebuild_eval.py` first |
| 401/403 from HuggingFace | Run `huggingface-cli login` again |
| ModuleNotFoundError: config | You're in the wrong directory. `cd /home/ubuntu/experiment/scripts` |
| ModuleNotFoundError: assistant_axis | Run `export ASSISTANT_AXIS_PATH=/home/ubuntu/assistant-axis` |
