"""
NaturalQuestions (NQ) Benchmark Runner
=======================================
Evaluates model on Google NaturalQuestions dataset.
Supports both retrieval (with context) and non-retrieval setups.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import NQ_DIR, NQ_DATASET, NQ_MAX_SAMPLES, RESULTS_DIR
from benchmark_utils import load_model, run_benchmark


def load_nq(max_samples=None):
    cache = NQ_DIR / "nq_validation"
    try:
        if cache.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(cache))
            print(f"  Loaded NQ from cache: {len(ds)} samples")
        else:
            from datasets import load_dataset
            print(f"  Downloading NQ from {NQ_DATASET}...")
            ds = load_dataset(NQ_DATASET, split="validation", trust_remote_code=True)
            NQ_DIR.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache))
            print(f"  Saved {len(ds)} validation samples")

        samples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = item.get("question", {})
            if isinstance(question, dict):
                question_text = question.get("text", "")
            else:
                question_text = str(question)

            annotations = item.get("annotations", {})
            if isinstance(annotations, dict):
                short_answers = annotations.get("short_answers", [])
            elif isinstance(annotations, list) and len(annotations) > 0:
                short_answers = annotations[0].get("short_answers", [])
            else:
                short_answers = []

            answer_texts = []
            for sa in short_answers:
                if isinstance(sa, dict) and "text" in sa:
                    answer_texts.append(sa["text"])
                elif isinstance(sa, list):
                    for span in sa:
                        if isinstance(span, dict) and "text" in span:
                            answer_texts.append(span["text"])

            answer = answer_texts[0] if answer_texts else ""
            if not answer:
                continue

            long_answer = item.get("long_answer", "")
            context_pages = []
            if isinstance(long_answer, str) and len(long_answer) > 50:
                clean = long_answer.replace("<P>", "").replace("</P>", "")
                clean = clean.replace("<Table>", "").replace("</Table>", "")
                clean = clean.replace("<Tr>", "").replace("</Tr>", "")
                clean = clean.replace("<Td>", "").replace("</Td>", "")
                clean = clean.replace("<Ul>", "").replace("</Ul>", "")
                clean = clean.replace("<Li>", "").replace("</Li>", "")
                clean = clean.replace("<H3>", "").replace("</H3>", "")
                clean = clean.replace("<Br>", "").replace("</Br>", "")
                if len(clean.strip()) > 50:
                    context_pages.append(clean[:2000])

            s = {
                "question": question_text,
                "answer": answer,
                "domain": "nq",
                "context": context_pages,
            }
            if s["question"] and s["answer"]:
                samples.append(s)

        print(f"NQ: {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"NQ failed ({e})")
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--both", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    max_s = args.max_samples or NQ_MAX_SAMPLES
    samples = load_nq(max_s)
    if not samples:
        print("No NQ samples loaded!")
        return

    if args.both:
        bm, bt = load_model(label="Base Mistral-7B")
        br = run_benchmark(bm, bt, samples, "Base Mistral-7B", "NQ")
        del bm, bt; torch.cuda.empty_cache()

        fm, ft = load_model(args.model, "AnchorGRPO Mistral-7B")
        fr = run_benchmark(fm, ft, samples, "AnchorGRPO Mistral-7B", "NQ")
        del fm, ft; torch.cuda.empty_cache()

        with open(RESULTS_DIR / "nq_base.json", "w") as f:
            json.dump(br["summary"], f, indent=2)
        with open(RESULTS_DIR / "nq_finetuned.json", "w") as f:
            json.dump(fr["summary"], f, indent=2)

        comp = {
            "base": br["summary"],
            "anchor_grpo": fr["summary"],
            "delta": {
                k: fr["summary"][k] - br["summary"][k]
                for k in ["accuracy", "hallucination_rate", "refusal_rate", "truthfulness"]
            },
        }
        out = args.output or str(RESULTS_DIR / "nq_comparison.json")
        with open(out, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"\nSaved: {out}")
    else:
        mdl, tok = load_model(args.model)
        res = run_benchmark(mdl, tok, samples, dataset_name="NQ")
        del mdl, tok; torch.cuda.empty_cache()
        out = args.output or str(RESULTS_DIR / "nq_results.json")
        with open(out, "w") as f:
            json.dump(res["summary"], f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    import torch
    main()
