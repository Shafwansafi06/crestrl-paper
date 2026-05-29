# ──────────────────────────────────────────────────────────────────────────────
# AnchorGRPO Setup — Windows PowerShell
# Quadro RTX 5000/6000 (16GB VRAM) via AnyDesk
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$WORK = Join-Path $DIR "workdir"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AnchorGRPO Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Force -Path $WORK | Out-Null

# ─── GPU ──────────────────────────────────────────────────────────────────────
Write-Host "[1/6] GPU..." -ForegroundColor Yellow
try { & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader } catch {
    Write-Host "  nvidia-smi not found!" -ForegroundColor Red; pause; exit 1
}
Write-Host ""

# ─── Python ───────────────────────────────────────────────────────────────────
Write-Host "[2/6] Python..." -ForegroundColor Yellow
$py = $null
foreach ($cmd in @("python3", "python", "py -3")) {
    try {
        $v = & cmd /c "$cmd --version" 2>&1
        if ($v -match "Python 3\.") { $py = $cmd; Write-Host "  $v" -ForegroundColor Green; break }
    } catch {}
}
if (-not $py) {
    Write-Host "  Python not found. Installing via winget..." -ForegroundColor Yellow
    try { & winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements; $py = "python" }
    catch { Write-Host "  Install Python from python.org"; pause; exit 1 }
}

# ─── Venv ─────────────────────────────────────────────────────────────────────
Write-Host "[3/6] Venv..." -ForegroundColor Yellow
$VENV = Join-Path $WORK ".venv"
& $py -m venv $VENV
. (Join-Path $VENV "Scripts\Activate.ps1")
python -m pip install --upgrade pip setuptools wheel -q
Write-Host "  Done" -ForegroundColor Green

# ─── PyTorch + CUDA 11.8 (matches driver 11.2) ───────────────────────────────
Write-Host "[4/6] PyTorch + CUDA 11.8..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
python -c "import torch; v=torch.cuda.is_available(); print(f'  PyTorch {torch.__version__}, CUDA {v}'); props=torch.cuda.get_device_properties(0); mem=getattr(props,'total_mem',getattr(props,'total_memory',0)); print(f'  {torch.cuda.get_device_name(0)} ({mem/1024**3:.0f}GB)') if v else print('  WARNING: CUDA not available')"

# ─── ML packages ──────────────────────────────────────────────────────────────
Write-Host "[5/6] ML dependencies..." -ForegroundColor Yellow
pip install "transformers>=4.40.0" -q
pip install "peft>=0.10.0" -q
pip install "trl>=0.12.0" -q
pip install "datasets>=2.18.0" -q
pip install "accelerate>=0.28.0" -q
pip install "bitsandbytes>=0.43.0" -q
pip install sentencepiece protobuf scipy scikit-learn numpy matplotlib seaborn pandas tabulate -q
pip install huggingface_hub safetensors -q
pip install rank_bm25 sentence_transformers -q
Write-Host "  Done" -ForegroundColor Green

# ─── Model ────────────────────────────────────────────────────────────────────
Write-Host "[6/6] Downloading Mistral-7B-Instruct-v0.3 (~14GB)..." -ForegroundColor Yellow
$modelDir = Join-Path $WORK "model_cache"
$modelDir = Join-Path $modelDir "Mistral-7B-Instruct-v0.3"
python -c "from huggingface_hub import snapshot_download; import os; d=r'$modelDir'; os.makedirs(d,exist_ok=True); snapshot_download('mistralai/Mistral-7B-Instruct-v0.3',local_dir=d,local_dir_use_symlinks=False); print('  Done.')"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File run_all.ps1" -ForegroundColor Green
Write-Host ""
pause
