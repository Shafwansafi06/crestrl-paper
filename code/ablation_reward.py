"""
Reward Component Ablation
==========================
Generates fresh responses on the GPU machine, scores each under four reward
configurations, and measures which components actually contribute signal.

Four configurations:
  R1  outcome only
  R2  outcome + calibration
  R3  outcome + anchor
  R4  full CrestRL (outcome + calibration + anchor)  [= compute_live_reward]

Uses two sample pools to cover all three verdict types:
  - NQ validation (EXISTS)      -> correct / abstain verdicts
  - Adversarial benchmark       -> hallucination / correct / abstain verdicts

Metric: Pearson r of each reward config vs ground-truth correctness label.
A config that adds signal should have higher |r| than the config below it.

Run on remote machine:
    python ablation_reward.py --nq 150 --adv 60 --out results/ablation_reward.json
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from benchmark_utils import load_model, evaluate_response, NO_CONTEXT_PROMPT
from run_nq_benchmark import load_nq
from reward import (
    get_verdict, outcome_reward, calibration_reward, anchor_reward,
    compute_p_know_from_logits, BENCHMARK,
)

LAMBDA_CALIB  = 0.22
LAMBDA_ANCHOR = 0.4
EPS_FLOOR     = 0.05


def generate(model, tokenizer, question: str, max_new_tokens=80) -> str:
    prompt = NO_CONTEXT_PROMPT.format(query=question)
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score_all_configs(verdict, p_know):
    """Return rewards for R1..R4 given a verdict and p_know."""
    r_out = outcome_reward(verdict, p_know)
    r_cal = calibration_reward(verdict, p_know, LAMBDA_CALIB)   # conf=p_know (validated)
    r_anc = anchor_reward(verdict, p_know)

    r1 = float(np.clip(r_out + EPS_FLOOR, -2.5, 1.5))
    r2 = float(np.clip(r_out + 0.16 * r_cal + EPS_FLOOR, -2.5, 1.5))
    r3 = float(np.clip(r_out + LAMBDA_ANCHOR * r_anc + EPS_FLOOR, -2.5, 1.5))
    r4 = float(np.clip(0.83 * r_out + LAMBDA_ANCHOR * r_anc + 0.16 * r_cal + EPS_FLOOR, -2.5, 1.5))

    return {"r_outcome": r_out, "r_calib": r_cal, "r_anchor": r_anc,
            "R1": r1, "R2": r2, "R3": r3, "R4": r4}


def run_samples(model, tokenizer, samples, source_label):
    records = []
    for i, s in enumerate(samples):
        q = s["question"]
        expected = s.get("expected", "EXISTS")
        gold = s.get("answer", "")

        response = generate(model, tokenizer, q)

        # Ground-truth correctness
        if gold:
            ev = evaluate_response(response, gold)
            gt_correct = int(ev["correct"])
            gt_refused = int(ev["refused"])
        else:
            # Adversarial: NOT_EXISTS — correct = abstained or refused
            gt_refused = int(is_abstaining(response))
            gt_correct = gt_refused  # for NOT_EXISTS, abstaining IS correct

        # Verdict for reward (uses expected label)
        verdict = get_verdict(response, expected)

        # p_know from logits
        p_know = compute_p_know_from_logits(model, tokenizer, q, response, model.device)

        scores = score_all_configs(verdict, p_know)

        records.append({
            "source": source_label,
            "question": q,
            "expected": expected,
            "gold": gold,
            "response": response,
            "verdict": verdict,
            "p_know": p_know,
            "gt_correct": gt_correct,
            "gt_refused": gt_refused,
            **scores,
        })

        torch.cuda.empty_cache()

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{source_label}] [{i+1}/{len(samples)}] verdict={verdict} "
                  f"p_know={p_know:.3f} gt={gt_correct} R4={scores['R4']:.3f}")

    return records


def is_abstaining(response):
    from reward import is_abstention
    return is_abstention(response)


def correlation_table(records):
    gt = np.array([r["gt_correct"] for r in records], dtype=float)
    rows = []
    for config in ["R1", "R2", "R3", "R4"]:
        vals = np.array([r[config] for r in records], dtype=float)
        corr, pval = pearsonr(vals, gt)
        rows.append({"config": config, "pearson_r": round(corr, 4), "p_value": round(pval, 4)})
    return rows


def verdict_breakdown(records):
    from collections import Counter
    counts = Counter(r["verdict"] for r in records)
    total = len(records)
    return {v: {"count": c, "pct": round(c / total * 100, 1)} for v, c in counts.items()}


def mean_reward_by_verdict(records):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        buckets[r["verdict"]].append(r["R4"])
    return {v: round(float(np.mean(vals)), 4) for v, vals in buckets.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq",  type=int, default=150,
                        help="NQ samples (EXISTS case, default 150)")
    parser.add_argument("--adv", type=int, default=60,
                        help="Adversarial benchmark samples (NOT_EXISTS + EXISTS, default 60)")
    parser.add_argument("--out", type=str,
                        default=str(SCRIPT_DIR / "results" / "ablation_reward.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("Reward Component Ablation")
    print("=" * 60)

    model, tokenizer = load_model()

    # ── NQ samples (EXISTS) ───────────────────────────────────────────────────
    nq_raw = load_nq(max_samples=args.nq * 4)   # over-fetch; NQ is sparse
    nq_samples = [{"question": s["question"], "answer": s["answer"], "expected": "EXISTS"}
                  for s in nq_raw][:args.nq]
    print(f"\nNQ pool: {len(nq_samples)} samples")

    # ── Adversarial benchmark (mixed EXISTS / NOT_EXISTS) ─────────────────────
    import random
    random.seed(42)
    adv_pool = [{"question": q, "answer": "", "expected": e}
                for _, _, q, e in BENCHMARK]
    adv_samples = random.sample(adv_pool, min(args.adv, len(adv_pool)))
    print(f"Adversarial pool: {len(adv_samples)} samples")

    # ── Run inference + scoring ───────────────────────────────────────────────
    print("\n--- NQ (EXISTS) ---")
    nq_records = run_samples(model, tokenizer, nq_samples, "nq")

    print("\n--- Adversarial ---")
    adv_records = run_samples(model, tokenizer, adv_samples, "adversarial")

    all_records = nq_records + adv_records

    # ── Analysis ──────────────────────────────────────────────────────────────
    corr_all  = correlation_table(all_records)
    corr_nq   = correlation_table(nq_records)
    corr_adv  = correlation_table(adv_records)
    verdicts  = verdict_breakdown(all_records)
    mean_r4   = mean_reward_by_verdict(all_records)

    print("\n" + "=" * 60)
    print("ABLATION TABLE — Pearson r (reward vs ground-truth correct)")
    print("=" * 60)
    labels = {
        "R1": "Outcome only              ",
        "R2": "Outcome + Calibration     ",
        "R3": "Outcome + Anchor          ",
        "R4": "Full CrestRL (R1+R2+R3)   ",
    }
    print(f"  {'Config':<32} {'All':>8} {'NQ':>8} {'Adv':>8}")
    print(f"  {'-'*56}")
    for row_all, row_nq, row_adv in zip(corr_all, corr_nq, corr_adv):
        c = row_all["config"]
        print(f"  {labels[c]} {row_all['pearson_r']:>8.4f} "
              f"{row_nq['pearson_r']:>8.4f} {row_adv['pearson_r']:>8.4f}")

    print(f"\n  Verdict breakdown (n={len(all_records)}):")
    for v, d in verdicts.items():
        print(f"    {v:<15} {d['count']:>4}  ({d['pct']:.1f}%)  "
              f"mean R4={mean_r4.get(v, 'n/a')}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "n_nq": len(nq_records),
        "n_adv": len(adv_records),
        "n_total": len(all_records),
        "correlation_all":  corr_all,
        "correlation_nq":   corr_nq,
        "correlation_adv":  corr_adv,
        "verdict_breakdown": verdicts,
        "mean_R4_by_verdict": mean_r4,
        "records": all_records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
