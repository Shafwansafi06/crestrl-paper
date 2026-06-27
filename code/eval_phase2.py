"""
Phase 2 Evaluation — Qwen2.5-1.5B: Base vs CrestRL vs TruthRL
==============================================================
Runs three models on NQ, HotpotQA, CRAG (MuSiQue excluded — retrieval artefact).
Saves results to phase2_<dataset>_<model>.json, does not touch Phase 1 results.

Usage:
    python eval_phase2.py \
        --base   Qwen/Qwen2.5-1.5B-Instruct \
        --crestrl <path_to_merged_crestrl> \
        --truthrl <path_to_merged_truthrl> \
        --max-samples 500

All three --model flags are optional; omit any to skip that model.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch
from config import RESULTS_DIR
from benchmark_utils import load_model, run_benchmark


DATASETS = {
    "nq":       ("NQ",       "run_nq_benchmark",       "load_nq"),
    "hotpotqa": ("HotpotQA", "run_hotpotqa_benchmark",  "load_hotpotqa"),
    "crag":     ("CRAG",     "run_crag_benchmark",      "load_crag"),
}


def load_dataset(key, max_samples):
    label, module, fn = DATASETS[key]
    mod = __import__(module, fromlist=[fn])
    samples = getattr(mod, fn)(max_samples)
    return label, samples


def run_one(model_path, model_label, dataset_label, samples, out_path):
    if out_path.exists():
        print(f"  [{model_label}] {dataset_label}: result exists, skipping.")
        return json.load(open(out_path))

    print(f"\n  Loading {model_label}...")
    mdl, tok = load_model(model_path, model_label)
    result = run_benchmark(mdl, tok, samples, model_label, dataset_label)
    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(result["summary"], f, indent=2)
    print(f"  Saved: {out_path}")
    return result["summary"]


def comparison_table(results: dict, dataset_label: str):
    """Print and return a 3-model comparison table."""
    metrics = ["accuracy", "hallucination_rate", "refusal_rate", "truthfulness"]
    print(f"\n  {'Metric':<22}", end="")
    for label in results:
        print(f"  {label:>18}", end="")
    print()
    print("  " + "-" * (22 + 20 * len(results)))

    rows = {}
    for metric in metrics:
        print(f"  {metric:<22}", end="")
        for label, summary in results.items():
            v = summary.get(metric, 0)
            print(f"  {v:>17.1f}%", end="")
        rows[metric] = {label: summary.get(metric, 0) for label, summary in results.items()}
    print()

    # Delta: CrestRL vs Base, TruthRL vs Base
    labels = list(results.keys())
    if len(labels) >= 2:
        base_label = labels[0]
        print(f"\n  Deltas vs {base_label}:")
        for label in labels[1:]:
            print(f"    {label}:")
            for metric in metrics:
                base_v = results[base_label].get(metric, 0)
                ft_v   = results[label].get(metric, 0)
                print(f"      {metric:<22} {ft_v - base_v:>+.1f}pp")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",     default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Base model id or path")
    p.add_argument("--crestrl",  default=None, help="Path to merged CrestRL model")
    p.add_argument("--truthrl",  default=None, help="Path to merged TruthRL model")
    p.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "crag"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--max-samples", type=int, default=500)
    args = p.parse_args()

    models = {"Qwen-base": args.base}
    if args.crestrl:
        models["CrestRL"] = args.crestrl
    if args.truthrl:
        models["TruthRL"] = args.truthrl

    print("=" * 60)
    print("PHASE 2 EVALUATION")
    print(f"  Models:   {list(models.keys())}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Max samples: {args.max_samples}")
    print("=" * 60)

    all_comparisons = {}
    t0 = time.time()

    for ds_key in args.datasets:
        ds_label, samples = load_dataset(ds_key, args.max_samples)
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_label}  ({len(samples)} samples)")
        print(f"{'='*60}")

        ds_results = {}
        for model_label, model_path in models.items():
            safe_label = model_label.lower().replace(" ", "_").replace("/", "_")
            out_path = RESULTS_DIR / f"phase2_{ds_key}_{safe_label}.json"
            summary = run_one(model_path, model_label, ds_label, samples, out_path)
            ds_results[model_label] = summary

        print(f"\n  --- {ds_label} comparison ---")
        rows = comparison_table(ds_results, ds_label)
        all_comparisons[ds_key] = {
            "dataset": ds_label,
            "results": {k: v for k, v in ds_results.items()},
            "deltas": rows,
        }

    # Save combined comparison
    comp_path = RESULTS_DIR / "phase2_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE — {elapsed/60:.1f} min")
    print(f"Combined comparison: {comp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
