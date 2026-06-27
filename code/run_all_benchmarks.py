"""
Run All Benchmarks — Unified Runner
====================================
Runs base and finetuned models on all 4 benchmarks (CRAG, NQ, HotpotQA, MuSiQue).
Optimized for Quadro RTX 5000 (16GB VRAM) — one model at a time.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch

from config import RESULTS_DIR
from benchmark_utils import load_model, run_benchmark


def run_benchmark_suite(dataset_name, samples, model_path=None, max_samples=None):
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"\n{'#'*60}")
    print(f"  {dataset_name} — {len(samples)} samples")
    print(f"{'#'*60}")

    if not samples:
        print(f"  No samples for {dataset_name}, skipping")
        return

    base_file = RESULTS_DIR / f"{dataset_name.lower()}_base.json"
    ft_file = RESULTS_DIR / f"{dataset_name.lower()}_finetuned.json"
    comp_file = RESULTS_DIR / f"{dataset_name.lower()}_comparison.json"

    # ─── Base Model ───────────────────────────────────────────────────────────
    if not base_file.exists():
        bm, bt = load_model(label="Base Mistral-7B")
        br = run_benchmark(bm, bt, samples, "Base Mistral-7B", dataset_name)
        del bm, bt; gc.collect(); torch.cuda.empty_cache()
        with open(base_file, "w") as f:
            json.dump(br["summary"], f, indent=2)
        print(f"  Saved: {base_file}")
    else:
        print(f"  Base results exist, skipping base run")
        br = {"summary": json.load(open(base_file))}

    # ─── Finetuned Model (skip if not available) ─────────────────────────────
    if not ft_file.exists():
        # Check if model path is valid before trying to load
        has_model = False
        if model_path:
            from pathlib import Path as _P
            mp = _P(model_path)
            # Check merged model or checkpoints
            merged_cfg = mp / "config.json" if mp.exists() else None
            ckpt_cfg = RESULTS_DIR.parent / "checkpoints" / "adapter_config.json"
            has_model = (merged_cfg and merged_cfg.exists()) or ckpt_cfg.exists()

        if has_model:
            fm, ft = load_model(model_path, "AnchorGRPO Mistral-7B")
            fr = run_benchmark(fm, ft, samples, "AnchorGRPO Mistral-7B", dataset_name)
            del fm, ft; gc.collect(); torch.cuda.empty_cache()
            with open(ft_file, "w") as f:
                json.dump(fr["summary"], f, indent=2)
            print(f"  Saved: {ft_file}")
        else:
            print(f"  No finetuned model found — skipping finetuned evaluation")
            print(f"  To run: train the model first with: python train.py --step data && python train.py --step train && python train.py --step merge")
            return


def main():
    p = argparse.ArgumentParser(description="Run all TruthRL benchmarks")
    p.add_argument("--model", default=None, help="Path to finetuned model")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    p.add_argument("--datasets", nargs="+", default=["crag", "nq", "hotpotqa", "musique"],
                   choices=["crag", "nq", "hotpotqa", "musique"], help="Which datasets to run")
    args = p.parse_args()

    print("=" * 60)
    print("RUN ALL BENCHMARKS")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model or 'default'}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 60)

    t0 = time.time()

    if "crag" in args.datasets:
        from run_crag_benchmark import load_crag
        crag_max = args.max_samples if args.max_samples else None
        samples = load_crag(crag_max)
        run_benchmark_suite("CRAG", samples, args.model, args.max_samples)

    if "nq" in args.datasets:
        from run_nq_benchmark import load_nq
        samples = load_nq(args.max_samples)
        run_benchmark_suite("NQ", samples, args.model, args.max_samples)

    if "hotpotqa" in args.datasets:
        from run_hotpotqa_benchmark import load_hotpotqa
        samples = load_hotpotqa(args.max_samples)
        run_benchmark_suite("HotpotQA", samples, args.model, args.max_samples)

    if "musique" in args.datasets:
        from run_musique_benchmark import load_musique
        samples = load_musique(args.max_samples)
        run_benchmark_suite("MuSiQue", samples, args.model, args.max_samples)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL BENCHMARKS COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"Results in: {RESULTS_DIR}")

    for f in sorted(RESULTS_DIR.glob("*_comparison.json")):
        data = json.load(open(f))
        d = data.get("delta", {})
        print(f"  {f.name}: acc={d.get('accuracy',0):+.1f}% "
              f"halluc={d.get('hallucination_rate',0):+.1f}% "
              f"truth={d.get('truthfulness',0):+.1f}%")


if __name__ == "__main__":
    main()
