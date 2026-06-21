"""
p_know Proxy Validation
========================
Tests whether inverse token perplexity of the gold answer given a prompt
correlates with the model's actual correctness on that question.

This measures the proxy on Mistral-7B itself (not Kadavath et al.'s model).
Run on the remote machine with GPU.

Output: Pearson r, p-value, 95% CI, and results JSON.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, bootstrap

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from benchmark_utils import load_model, evaluate_response, NO_CONTEXT_PROMPT
from run_nq_benchmark import load_nq


def gold_answer_perplexity(model, tokenizer, question: str, gold: str) -> float:
    """
    Compute per-token cross-entropy loss of `gold` tokens conditioned on `question`.
    Low PPL = model assigns high probability to this answer = high self-knowledge.
    """
    prompt = NO_CONTEXT_PROMPT.format(query=question)
    full_text = prompt + " " + gold

    enc_prompt = tokenizer(prompt, return_tensors="pt").input_ids
    enc_full = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=512).input_ids

    prompt_len = enc_prompt.shape[1]
    answer_len = enc_full.shape[1] - prompt_len
    if answer_len <= 0:
        return float("inf")

    enc_full = enc_full.to(model.device)
    with torch.no_grad():
        out = model(enc_full)
        logits = out.logits[0]  # [seq_len, vocab]

    # Loss only over answer tokens (shifted by 1)
    shift_logits = logits[prompt_len - 1 : -1]      # [answer_len, vocab]
    shift_labels = enc_full[0, prompt_len:]          # [answer_len]

    loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction="mean"
    )
    return float(torch.exp(loss).item())


def generate_response(model, tokenizer, question: str) -> str:
    prompt = NO_CONTEXT_PROMPT.format(query=question)
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=80, temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def bootstrap_ci(x, y, n_resamples=1000, confidence=0.95):
    """95% CI for Pearson r via bootstrap."""
    def r_stat(a, b):
        return pearsonr(a, b)[0]

    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(x), size=len(x))
        try:
            boots.append(r_stat(x[idx], y[idx]))
        except Exception:
            pass
    lo = np.percentile(boots, (1 - confidence) / 2 * 100)
    hi = np.percentile(boots, (1 + confidence) / 2 * 100)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300,
                        help="Number of NQ samples to evaluate (default: 300)")
    parser.add_argument("--out", type=str, default=str(SCRIPT_DIR / "results" / "p_know_validation.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("p_know Proxy Validation on Mistral-7B / NQ")
    print("=" * 60)

    model, tokenizer = load_model()
    samples = load_nq(max_samples=args.n)

    ppls, correct_flags = [], []
    records = []

    for i, s in enumerate(samples):
        q, gold = s["question"], s["answer"]

        ppl = gold_answer_perplexity(model, tokenizer, q, gold)
        response = generate_response(model, tokenizer, q)
        ev = evaluate_response(response, gold)
        correct = ev["correct"]

        ppls.append(ppl)
        correct_flags.append(int(correct))
        records.append({
            "question": q, "gold": gold,
            "response": response, "ppl": ppl, "correct": correct,
        })

        torch.cuda.empty_cache()

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] ppl={ppl:.1f} correct={correct}  q={q[:50]}...")

    # ── Filter extreme PPL outliers before correlation ────────────────────────
    ppls = np.array(ppls, dtype=float)
    correct_flags = np.array(correct_flags, dtype=float)

    valid = np.isfinite(ppls) & (ppls < 1e4)
    n_valid = valid.sum()
    if n_valid < 20:
        print(f"Only {n_valid} valid samples — cannot compute reliable correlation.")
        return

    x = 1.0 / ppls[valid]      # inverse PPL as p_know proxy
    y = correct_flags[valid]

    r, p = pearsonr(x, y)
    ci_lo, ci_hi = bootstrap_ci(x, y)

    ppl_correct = ppls[valid & (correct_flags == 1)]
    ppl_wrong   = ppls[valid & (correct_flags == 0)]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated:     {len(samples)}")
    print(f"  Valid (PPL < 10k):     {n_valid}")
    print(f"  Model accuracy:        {y.mean() * 100:.1f}%")
    print(f"  Pearson r (1/PPL vs correct): {r:.4f}  (p={p:.4f})")
    print(f"  95% bootstrap CI:      [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Mean PPL (correct):    {ppl_correct.mean():.1f}  (n={len(ppl_correct)})")
    print(f"  Mean PPL (incorrect):  {ppl_wrong.mean():.1f}  (n={len(ppl_wrong)})")
    print("=" * 60)

    if r > 0.3 and p < 0.05:
        verdict = "PROXY VALID — use in CrestRL reward"
    elif r > 0.1 and p < 0.05:
        verdict = "WEAK CORRELATION — proxy is marginal, needs reframing"
    else:
        verdict = "PROXY INVALID — switch to Path B (reframe as metrics paper)"
    print(f"\n  Verdict: {verdict}")

    out = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "dataset": "NQ validation",
        "n_samples": len(samples),
        "n_valid": int(n_valid),
        "accuracy": float(y.mean()),
        "pearson_r": float(r),
        "p_value": float(p),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "mean_ppl_correct": float(ppl_correct.mean()) if len(ppl_correct) else None,
        "mean_ppl_incorrect": float(ppl_wrong.mean()) if len(ppl_wrong) else None,
        "verdict": verdict,
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
