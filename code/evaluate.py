"""
Head-to-Head Evaluation: Binary vs TruthRL vs CrestRL V2 vs AnchorGRPO
========================================================================

Metrics:
- Accuracy: % correct
- Hallucination Rate: % hallucinated on fake entities
- False Positive Rate: % wrong refusals on real entities
- Truthfulness: accuracy + abstention - hallucination (CRAG metric)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_CACHE_DIR, BASE_MODEL, RESULTS_DIR, EPS_FLOOR, LAMBDA_CALIB, ALPHA_ANCHOR
from reward import (
    get_verdict, extract_confidence, BENCHMARK,
    outcome_reward, calibration_reward, anchor_reward,
    compute_group_advantages, compute_p_know_from_logits as p_know_from_logits,
)


# ─── Reward Functions (all 4 methods) ─────────────────────────────────────────

def binary_reward(r, e):
    v = get_verdict(r, e)
    if v == "correct": return 1.0
    if v == "abstain": return 0.0
    return -1.0

def truthrl_reward(r, e):
    v = get_verdict(r, e)
    if v == "correct": return 1.0
    if v == "abstain": return 0.0
    return -1.0

def crest_v2_reward(r, e, p_know):
    v = get_verdict(r, e)
    conf = extract_confidence(r)
    r_out = outcome_reward(v, p_know)
    r_cal = calibration_reward(v, conf, LAMBDA_CALIB)
    r_anc = anchor_reward(v, p_know)
    total = 0.83 * r_out + ALPHA_ANCHOR * r_anc + 0.16 * r_cal + EPS_FLOOR
    return float(np.clip(total, -2.5, 1.5))

def anchor_grpo_reward(r, e, p_know):
    return crest_v2_reward(r, e, p_know)


# ─── Reward Functions (all 4 methods) ─────────────────────────────────────────

def binary_reward(r, e):
    v = get_verdict(r, e)
    if v == "correct": return 1.0
    if v == "abstain": return 0.0
    return -1.0

def truthrl_reward(r, e):
    v = get_verdict(r, e)
    if v == "correct": return 1.0
    if v == "abstain": return 0.0
    return -1.0

def crest_v2_reward(r, e, p_know):
    v = get_verdict(r, e)
    conf = extract_confidence(r)
    r_out = outcome_reward(v, p_know)
    r_cal = calibration_reward(v, conf, LAMBDA_CALIB)
    r_anc = anchor_reward(v, p_know)
    total = 0.83 * r_out + ALPHA_ANCHOR * r_anc + 0.16 * r_cal + EPS_FLOOR
    return float(np.clip(total, -2.5, 1.5))

def anchor_grpo_reward(r, e, p_know):
    return crest_v2_reward(r, e, p_know)


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(path=None):
    p = path or str(MODEL_CACHE_DIR / "Mistral-7B-Instruct-v0.3")
    if not Path(p).exists(): p = BASE_MODEL
    print(f"Loading: {p}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return mdl, tok


def generate(mdl, tok, prompt):
    msgs = [{"role": "system", "content": "You are a helpful, accurate assistant. If unsure, say so. Do not fabricate."},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=300, temperature=0.3, top_p=0.9,
                           do_sample=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(mdl, tok):
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD EVALUATION")
    print(f"Benchmark: {len(BENCHMARK)} prompts")
    print(f"{'='*60}")

    methods = ["binary", "truthrl", "crestrl_v2"]
    results = {m: {"rewards": [], "correct": 0, "hallucinated": 0, "fp": 0, "abstained": 0}
               for m in methods}
    per_cat = {}
    t0 = time.time()

    for i, (cid, cat, prompt, expected) in enumerate(BENCHMARK):
        # Generate response
        response = generate(mdl, tok, prompt)
        torch.cuda.empty_cache()

        # Logit-based p_know
        p_know = p_know_from_logits(mdl, tok, prompt, response, mdl.device)

        verdict = get_verdict(response, expected)
        is_fake = expected == "NOT_EXISTS"

        # Compute rewards for each method
        bin_r = binary_reward(response, expected)
        trl_r = truthrl_reward(response, expected)
        cst_r = crest_v2_reward(response, expected, p_know)

        for m, r in [("binary", bin_r), ("truthrl", trl_r), ("crestrl_v2", cst_r)]:
            results[m]["rewards"].append(r)

        # Classification (same for all methods — based on verdict)
        for m in methods:
            if verdict == "correct":
                results[m]["correct"] += 1
            elif verdict == "abstain":
                results[m]["abstained"] += 1
                if not is_fake:
                    results[m]["fp"] += 1  # refusing a real entity
            else:  # hallucination
                results[m]["hallucinated"] += 1

        # Per-category
        if cat not in per_cat:
            per_cat[cat] = {"total": 0, "correct": 0, "hallucinated": 0, "fp": 0, "abstained": 0}
        per_cat[cat]["total"] += 1
        if verdict == "correct": per_cat[cat]["correct"] += 1
        elif verdict == "abstain": per_cat[cat]["abstained"] += 1; per_cat[cat]["fp"] += int(not is_fake)
        else: per_cat[cat]["hallucinated"] += 1

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(BENCHMARK)}] p_know={p_know:.3f} verdict={verdict} {time.time()-t0:.0f}s")

    # Compute final metrics
    n = len(results["binary"]["rewards"])
    elapsed = time.time() - t0

    output = {}
    for m in methods:
        d = results[m]
        r = d["rewards"]
        output[m] = {
            "avg_reward": float(np.mean(r)) if r else 0,
            "std_reward": float(np.std(r)) if r else 0,
            "accuracy": d["correct"] / n * 100 if n else 0,
            "hallucination_rate": d["hallucinated"] / n * 100 if n else 0,
            "false_positive_rate": d["fp"] / n * 100 if n else 0,
            "abstention_rate": d["abstained"] / n * 100 if n else 0,
            "truthfulness": (d["correct"] + d["abstained"] - d["hallucinated"]) / n * 100 if n else 0,
        }

    print(f"\n{'Metric':<25} {'Binary':>10} {'TruthRL':>10} {'CrestRL':>10}")
    print("-" * 58)
    for k in ["accuracy", "hallucination_rate", "false_positive_rate", "abstention_rate", "truthfulness", "avg_reward"]:
        label = k.replace("_", " ").title()
        vals = [output[m][k] for m in methods]
        if k == "avg_reward":
            print(f"{label:<25} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")
        else:
            print(f"{label:<25} {vals[0]:>9.1f}% {vals[1]:>9.1f}% {vals[2]:>9.1f}%")

    print(f"\n{'Category':<15} {'N':>5} {'Acc%':>7} {'Hall%':>7} {'FP%':>7} {'Truth%':>7}")
    print("-" * 50)
    for c in sorted(per_cat):
        s = per_cat[c]; t = max(s["total"], 1)
        truth = (s["correct"] + s["abstained"] - s["hallucinated"]) / t * 100
        print(f"{c:<15} {s['total']:>5} {s['correct']/t*100:>6.1f}% {s['hallucinated']/t*100:>6.1f}% {s['fp']/t*100:>6.1f}% {truth:>6.1f}%")

    return {"methods": output, "per_category": per_cat, "time_seconds": elapsed}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    mdl, tok = load_model(args.model)
    results = evaluate(mdl, tok)
    del mdl, tok; torch.cuda.empty_cache()

    out = args.output or str(RESULTS_DIR / "evaluation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
