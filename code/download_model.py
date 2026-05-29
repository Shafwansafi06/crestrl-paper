"""
Download Mistral-7B-Instruct-v0.3 — Standalone Script
Run this separately if setup.ps1 failed to download the model.

Usage:
    python download_model.py
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "workdir" / "model_cache" / "Mistral-7B-Instruct-v0.3"

print(f"Downloading to: {MODEL_DIR}")
print("This will take 15-30 minutes depending on internet speed.")
print("")

os.makedirs(MODEL_DIR, exist_ok=True)

from huggingface_hub import snapshot_download

snapshot_download(
    "mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)

# Verify
files = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
print(f"\nDownloaded {len(files)} model files")

if any(MODEL_DIR.glob("*.safetensors")):
    print("Model ready!")
else:
    print("WARNING: Model files may be incomplete. Check the directory.")
    print(f"  {MODEL_DIR}")
