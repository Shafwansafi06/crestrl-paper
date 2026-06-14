"""
MuSiQue Benchmark Runner
=========================
Evaluates model on MuSiQue multi-step compositional reasoning dataset.
2-4 hop questions over paragraph chains.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import MUSIQUE_DIR, MUSIQUE_DATASET, MUSIQUE_MAX_SAMPLES, RESULTS_DIR
from benchmark_utils import load_model, run_benchmark


def load_musique(max_samples=None):
    cache = MUSIQUE_DIR / "musique_validation"
    try:
        if cache.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(cache))
            print(f"  Loaded MuSiQue from cache: {len(ds)} samples")
        else:
            from datasets import load_dataset
            print(f"  Downloading MuSiQue from {MUSIQUE_DATASET}...")
            ds = load_dataset(MUSIQUE_DATASET, split="validation", trust_remote_code=True)
            MUSIQUE_DIR.mkdir(parents=True, exist_ok=True)
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

            context_pages = []
            paragraphs = item.get("paragraphs", [])
            if isinstance(paragraphs, list):
                for para in paragraphs:
                    if isinstance(para, dict):
                        title = para.get("title", "")
                        text = para.get("paragraph_text", para.get("text", ""))
                        if text:
                            context_pages.append(f"{title}: {text}" if title else text)
                    elif isinstance(para, str):
                        context_pages.append(para)

            s = {
                "question": question,
                "answer": answer,
                "domain": "musique",
                "context": context_pages,
            }
            samples.append(s)

        print(f"MuSiQue: {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"MuSiQue failed ({e})")
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--both", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    max_s = args.max_samples or MUSIQUE_MAX_SAMPLES
    samples = load_musique(max_s)
    if not samples:
        print("No MuSiQue samples loaded!")
        return

    if args.both:
        bm, bt = load_model(label="Base Mistral-7B")
        br = run_benchmark(bm, bt, samples, "Base Mistral-7B", "MuSiQue")
        del bm, bt; torch.cuda.empty_cache()

        fm, ft = load_model(args.model, "AnchorGRPO Mistral-7B")
        fr = run_benchmark(fm, ft, samples, "AnchorGRPO Mistral-7B", "MuSiQue")
        del fm, ft; torch.cuda.empty_cache()

        with open(RESULTS_DIR / "musique_base.json", "w") as f:
            json.dump(br["summary"], f, indent=2)
        with open(RESULTS_DIR / "musique_finetuned.json", "w") as f:
            json.dump(fr["summary"], f, indent=2)

        comp = {
            "base": br["summary"],
            "anchor_grpo": fr["summary"],
            "delta": {
                k: fr["summary"][k] - br["summary"][k]
                for k in ["accuracy", "hallucination_rate", "refusal_rate", "truthfulness"]
            },
        }
        out = args.output or str(RESULTS_DIR / "musique_comparison.json")
        with open(out, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"\nSaved: {out}")
    else:
        mdl, tok = load_model(args.model)
        res = run_benchmark(mdl, tok, samples, dataset_name="MuSiQue")
        del mdl, tok; torch.cuda.empty_cache()
        out = args.output or str(RESULTS_DIR / "musique_results.json")
        with open(out, "w") as f:
            json.dump(res["summary"], f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    import torch
    main()
