"""
Confidence Intervals — Accuracy and Inference Timing
======================================================
Two modes:

  --mode accuracy  (local, no GPU)
      Bootstraps 95% CIs on accuracy / refusal / hallucination rates
      from ablation_reward.json per-sample records.
      Also computes CIs for the adversarial benchmark by category.

  --mode timing    (remote GPU required)
      Runs inference on a fixed set of 50 NQ questions x 5 seeds,
      measures per-sample latency, reports mean +/- std and 95% CI.
      Compares base model vs finetuned to give a CI on the speedup claim.

Run locally:
    python compute_cis.py --mode accuracy

Run on remote:
    python compute_cis.py --mode timing
    python compute_cis.py --mode timing --model <path_to_finetuned>
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

ABLATION_PATH = SCRIPT_DIR / "results" / "ablation_reward.json"
N_BOOT = 2000
RNG = np.random.default_rng(42)


# ── Bootstrap utilities ───────────────────────────────────────────────────────

def bootstrap_ci(values, stat_fn=np.mean, n=N_BOOT, conf=0.95):
    values = np.asarray(values, dtype=float)
    boots = [stat_fn(RNG.choice(values, size=len(values), replace=True)) for _ in range(n)]
    lo = np.percentile(boots, (1 - conf) / 2 * 100)
    hi = np.percentile(boots, (1 + conf) / 2 * 100)
    return float(stat_fn(values)), float(lo), float(hi)


def fmt(mean, lo, hi, pct=True):
    scale = 100 if pct else 1
    return f"{mean*scale:.1f}% [{lo*scale:.1f}, {hi*scale:.1f}]"


# ── Mode: accuracy ────────────────────────────────────────────────────────────

def run_accuracy():
    with open(ABLATION_PATH) as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records from ablation_reward.json\n")

    def report_subset(label, subset):
        if not subset:
            print(f"  {label}: no data")
            return {}
        correct   = [r["gt_correct"] for r in subset]
        refused   = [r.get("gt_refused", int(r["verdict"] == "abstain")) for r in subset]
        hallu     = [int(r["verdict"] == "hallucination") for r in subset]

        m_c, lo_c, hi_c = bootstrap_ci(correct)
        m_r, lo_r, hi_r = bootstrap_ci(refused)
        m_h, lo_h, hi_h = bootstrap_ci(hallu)

        print(f"  {label}  (n={len(subset)})")
        print(f"    Accuracy:          {fmt(m_c, lo_c, hi_c)}")
        print(f"    Refusal rate:      {fmt(m_r, lo_r, hi_r)}")
        print(f"    Hallucination:     {fmt(m_h, lo_h, hi_h)}")
        return {
            "n": len(subset),
            "accuracy":       {"mean": round(m_c*100,1), "ci95": [round(lo_c*100,1), round(hi_c*100,1)]},
            "refusal_rate":   {"mean": round(m_r*100,1), "ci95": [round(lo_r*100,1), round(hi_r*100,1)]},
            "hallucination":  {"mean": round(m_h*100,1), "ci95": [round(lo_h*100,1), round(hi_h*100,1)]},
        }

    print("=" * 60)
    print("ACCURACY CIs  (95% bootstrap, n=2000 resamples)")
    print("=" * 60)

    nq_recs  = [r for r in records if r["source"] == "nq"]
    adv_recs = [r for r in records if r["source"] == "adversarial"]

    results = {}
    results["nq"]          = report_subset("NQ (EXISTS)",        nq_recs)
    print()
    results["adversarial"] = report_subset("Adversarial (mixed)", adv_recs)
    print()

    # Adversarial by category — from BENCHMARK structure
    from reward import BENCHMARK
    cat_map = {}
    for _, cat, q, _ in BENCHMARK:
        cat_map[q] = cat

    cats = {}
    for r in adv_recs:
        cat = cat_map.get(r["question"], "unknown")
        cats.setdefault(cat, []).append(r)

    if cats:
        print("  Adversarial by category:")
        results["by_category"] = {}
        for cat, recs in sorted(cats.items()):
            correct = [r["gt_correct"] for r in recs]
            m, lo, hi = bootstrap_ci(correct)
            print(f"    {cat:<15} n={len(recs):>3}  acc={fmt(m, lo, hi)}")
            results["by_category"][cat] = {
                "n": len(recs),
                "accuracy": {"mean": round(m*100,1), "ci95": [round(lo*100,1), round(hi*100,1)]}
            }

    # Reward correlation CI
    print("\n  Reward-correctness correlation (R4):")
    gt   = np.array([r["gt_correct"] for r in records], dtype=float)
    r4   = np.array([r["R4"] for r in records], dtype=float)
    r_val, p_val = pearsonr(r4, gt)
    boots_r = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(gt), size=len(gt))
        try:
            boots_r.append(pearsonr(r4[idx], gt[idx])[0])
        except Exception:
            pass
    r_lo = np.percentile(boots_r, 2.5)
    r_hi = np.percentile(boots_r, 97.5)
    print(f"    Pearson r = {r_val:.4f}  95% CI [{r_lo:.4f}, {r_hi:.4f}]  p={p_val:.4f}")
    results["reward_correlation"] = {
        "r": round(r_val, 4), "ci95": [round(r_lo, 4), round(r_hi, 4)], "p": round(p_val, 4)
    }

    out_path = SCRIPT_DIR / "results" / "ci_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return results


# ── Mode: timing ──────────────────────────────────────────────────────────────

def run_timing(model_path=None, n_samples=50, n_seeds=5):
    import torch
    from benchmark_utils import load_model, NO_CONTEXT_PROMPT
    from run_nq_benchmark import load_nq

    print("=" * 60)
    label = f"Finetuned ({model_path})" if model_path else "Base model"
    print(f"INFERENCE TIMING CI — {label}")
    print(f"  {n_samples} samples x {n_seeds} seeds")
    print("=" * 60)

    model, tokenizer = load_model(model_path)
    samples = load_nq(max_samples=n_samples * 3)[:n_samples]
    print(f"  Loaded {len(samples)} NQ samples\n")

    all_latencies = []  # shape: [n_seeds, n_samples]

    for seed in range(n_seeds):
        seed_latencies = []
        print(f"  Seed {seed+1}/{n_seeds}:")
        for i, s in enumerate(samples):
            prompt = NO_CONTEXT_PROMPT.format(query=s["question"])
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=100, temperature=0.1, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_out_tokens = out.shape[1] - inp["input_ids"].shape[1]
            seed_latencies.append({"total_s": elapsed, "tokens": n_out_tokens,
                                    "ms_per_token": elapsed * 1000 / max(n_out_tokens, 1)})
            torch.cuda.empty_cache()

        total_s = [x["total_s"] for x in seed_latencies]
        ms_tok  = [x["ms_per_token"] for x in seed_latencies]
        print(f"    mean={np.mean(total_s):.2f}s  std={np.std(total_s):.2f}s  "
              f"ms/tok={np.mean(ms_tok):.1f}")
        all_latencies.append(seed_latencies)

    # Flatten across seeds for CI
    flat_total = [x["total_s"] for seed in all_latencies for x in seed]
    flat_mstok = [x["ms_per_token"] for seed in all_latencies for x in seed]

    m_t, lo_t, hi_t = bootstrap_ci(flat_total)
    m_m, lo_m, hi_m = bootstrap_ci(flat_mstok)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total latency/sample:  {m_t:.3f}s  95% CI [{lo_t:.3f}, {hi_t:.3f}]")
    print(f"  ms per output token:   {m_m:.2f}  95% CI [{lo_m:.2f}, {hi_m:.2f}]")
    print(f"  Throughput:            {1/m_t:.2f} samples/s")

    result = {
        "model": label,
        "n_samples": n_samples,
        "n_seeds": n_seeds,
        "latency_per_sample": {"mean_s": round(m_t, 4),
                                "ci95": [round(lo_t, 4), round(hi_t, 4)]},
        "ms_per_token": {"mean": round(m_m, 2),
                         "ci95": [round(lo_m, 2), round(hi_m, 2)]},
        "throughput_samples_per_sec": round(1/m_t, 3),
    }

    tag = "finetuned" if model_path else "base"
    out_path = SCRIPT_DIR / "results" / f"ci_timing_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return result


# ── Compare timing results ────────────────────────────────────────────────────

def compare_timing():
    base_path = SCRIPT_DIR / "results" / "ci_timing_base.json"
    ft_path   = SCRIPT_DIR / "results" / "ci_timing_finetuned.json"
    if not base_path.exists() or not ft_path.exists():
        print("Run --mode timing for both base and finetuned first.")
        return

    with open(base_path)  as f: base = json.load(f)
    with open(ft_path)    as f: ft   = json.load(f)

    base_m = base["latency_per_sample"]["mean_s"]
    ft_m   = ft["latency_per_sample"]["mean_s"]
    delta_pct = (ft_m - base_m) / base_m * 100

    base_lo, base_hi = base["latency_per_sample"]["ci95"]
    ft_lo,   ft_hi   = ft["latency_per_sample"]["ci95"]

    print("\n" + "=" * 60)
    print("TIMING COMPARISON")
    print("=" * 60)
    print(f"  Base:      {base_m:.3f}s  [{base_lo:.3f}, {base_hi:.3f}]")
    print(f"  Finetuned: {ft_m:.3f}s  [{ft_lo:.3f}, {ft_hi:.3f}]")
    print(f"  Delta:     {delta_pct:+.1f}%")

    out = {"base": base, "finetuned": ft, "delta_pct": round(delta_pct, 1)}
    out_path = SCRIPT_DIR / "results" / "ci_timing_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["accuracy", "timing", "compare"],
                        default="accuracy")
    parser.add_argument("--model", default=None,
                        help="Path to finetuned model (timing mode only)")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-seeds",   type=int, default=5)
    args = parser.parse_args()

    if args.mode == "accuracy":
        run_accuracy()
    elif args.mode == "timing":
        run_timing(args.model, args.n_samples, args.n_seeds)
    elif args.mode == "compare":
        compare_timing()


if __name__ == "__main__":
    main()
