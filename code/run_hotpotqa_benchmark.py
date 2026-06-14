"""
HotpotQA Benchmark Runner
==========================
Evaluates model on HotpotQA multi-hop reasoning dataset.
Distractor split: 10 paragraphs (2 relevant + 8 distractors).
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import HOTPOTQA_DIR, HOTPOTQA_DATASET, HOTPOTQA_MAX_SAMPLES, RESULTS_DIR
from benchmark_utils import load_model, run_benchmark


def load_hotpotqa(max_samples=None):
    cache = HOTPOTQA_DIR / "hotpotqa_distractor_validation"
    try:
        if cache.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(cache))
            print(f"  Loaded HotpotQA from cache: {len(ds)} samples")
        else:
            from datasets import load_dataset
            print(f"  Downloading HotpotQA from {HOTPOTQA_DATASET}...")
            ds = load_dataset(HOTPOTQA_DATASET, "distractor", split="validation", trust_remote_code=True)
            HOTPOTQA_DIR.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache))
            print(f"  Saved {len(ds)} validation samples")

        samples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = item.get("question", "")
            answer = item.get("answer", "")
            if not question or not answer:
                continue

            context_paragraphs = item.get("context", [])
            context_pages = []
            if isinstance(context_paragraphs, list):
                for para in context_paragraphs:
                    if isinstance(para, list) and len(para) >= 2:
                        title = para[0] if isinstance(para[0], str) else ""
                        sentences = para[1]
                        if isinstance(sentences, list):
                            text = " ".join(sentences)
                        else:
                            text = str(sentences)
                        context_pages.append(f"{title}: {text}")
                    elif isinstance(para, dict):
                        title = para.get("title", "")
                        text = para.get("sentences", para.get("text", ""))
                        if isinstance(text, list):
                            text = " ".join(text)
                        context_pages.append(f"{title}: {text}")

            s = {
                "question": question,
                "answer": answer,
                "domain": "hotpotqa",
                "context": context_pages,
            }
            samples.append(s)

        print(f"HotpotQA: {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"HotpotQA failed ({e})")
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--both", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    max_s = args.max_samples or HOTPOTQA_MAX_SAMPLES
    samples = load_hotpotqa(max_s)
    if not samples:
        print("No HotpotQA samples loaded!")
        return

    if args.both:
        bm, bt = load_model(label="Base Mistral-7B")
        br = run_benchmark(bm, bt, samples, "Base Mistral-7B", "HotpotQA")
        del bm, bt; torch.cuda.empty_cache()

        fm, ft = load_model(args.model, "AnchorGRPO Mistral-7B")
        fr = run_benchmark(fm, ft, samples, "AnchorGRPO Mistral-7B", "HotpotQA")
        del fm, ft; torch.cuda.empty_cache()

        with open(RESULTS_DIR / "hotpotqa_base.json", "w") as f:
            json.dump(br["summary"], f, indent=2)
        with open(RESULTS_DIR / "hotpotqa_finetuned.json", "w") as f:
            json.dump(fr["summary"], f, indent=2)

        comp = {
            "base": br["summary"],
            "anchor_grpo": fr["summary"],
            "delta": {
                k: fr["summary"][k] - br["summary"][k]
                for k in ["accuracy", "hallucination_rate", "refusal_rate", "truthfulness"]
            },
        }
        out = args.output or str(RESULTS_DIR / "hotpotqa_comparison.json")
        with open(out, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"\nSaved: {out}")
    else:
        mdl, tok = load_model(args.model)
        res = run_benchmark(mdl, tok, samples, dataset_name="HotpotQA")
        del mdl, tok; torch.cuda.empty_cache()
        out = args.output or str(RESULTS_DIR / "hotpotqa_results.json")
        with open(out, "w") as f:
            json.dump(res["summary"], f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    import torch
    main()
