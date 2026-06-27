"""
GPU Diagnostic — Run this first to verify CUDA setup
"""
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_mem', getattr(props, 'total_memory', 0))
        print(f"VRAM: {mem / 1024**3:.1f}GB")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("NO GPU DETECTED!")
        print("Check: nvidia-smi")
        print("If GPU shows in nvidia-smi but not here, reinstall PyTorch with CUDA:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
except ImportError as e:
    print(f"Import error: {e}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__}")
except ImportError:
    print("bitsandbytes not installed!")

try:
    from transformers import AutoModelForCausalLM
    print(f"transformers: {AutoModelForCausalLM.__module__}")
except ImportError:
    print("transformers not installed!")
