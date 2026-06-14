"""
Dataset Downloader — NQ, HotpotQA, MuSiQue
============================================
Downloads and caches datasets from HuggingFace for TruthRL benchmark evaluation.
Runs on CPU only — no VRAM required.
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import NQ_DATASET, NQ_DIR, HOTPOTQA_DATASET, HOTPOTQA_DIR, MUSIQUE_DATASET, MUSIQUE_DIR


def download_nq():
    cache = NQ_DIR / "nq_validation"
    if cache.exists():
        print(f"  NQ already cached at {cache}")
        from datasets import load_from_disk
        ds = load_from_disk(str(cache))
        print(f"  Loaded {len(ds)} samples")
        return ds

    print(f"  Downloading NQ from {NQ_DATASET}...")
    from datasets import load_dataset
    ds = load_dataset(NQ_DATASET, split="validation", trust_remote_code=True)

    NQ_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    print(f"  Saved {len(ds)} samples to {cache}")
    return ds


def download_hotpotqa():
    cache = HOTPOTQA_DIR / "hotpotqa_distractor_validation"
    if cache.exists():
        print(f"  HotpotQA already cached at {cache}")
        from datasets import load_from_disk
        ds = load_from_disk(str(cache))
        print(f"  Loaded {len(ds)} samples")
        return ds

    print(f"  Downloading HotpotQA from {HOTPOTQA_DATASET}...")
    from datasets import load_dataset
    ds = load_dataset(HOTPOTQA_DATASET, "distractor", split="validation", trust_remote_code=True)

    HOTPOTQA_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    print(f"  Saved {len(ds)} samples to {cache}")
    return ds


def download_musique():
    cache = MUSIQUE_DIR / "musique_validation"
    if cache.exists():
        print(f"  MuSiQue already cached at {cache}")
        from datasets import load_from_disk
        ds = load_from_disk(str(cache))
        print(f"  Loaded {len(ds)} samples")
        return ds

    print(f"  Downloading MuSiQue from {MUSIQUE_DATASET}...")
    from datasets import load_dataset
    ds = load_dataset(MUSIQUE_DATASET, split="validation", trust_remote_code=True)

    MUSIQUE_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    print(f"  Saved {len(ds)} samples to {cache}")
    return ds


def main():
    p = argparse.ArgumentParser(description="Download TruthRL benchmark datasets")
    p.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "musique"],
                   choices=["nq", "hotpotqa", "musique"], help="Which datasets to download")
    args = p.parse_args()

    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)

    t0 = time.time()

    if "nq" in args.datasets:
        print("\n[1/3] NaturalQuestions:")
        download_nq()

    if "hotpotqa" in args.datasets:
        print("\n[2/3] HotpotQA:")
        download_hotpotqa()

    if "musique" in args.datasets:
        print("\n[3/3] MuSiQue:")
        download_musique()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE — {elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
