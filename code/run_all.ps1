# ──────────────────────────────────────────────────────────────────────────────
# AnchorGRPO Full Pipeline — Windows PowerShell
# Quadro RTX 5000 (16GB VRAM)
#
# Usage:
#   .\run_all.ps1                    # Run everything
#   .\run_all.ps1 -EvalOnly          # Skip training, just evaluate
#   .\run_all.ps1 -SkipTrain         # Skip training, run benchmarks
#   .\run_all.ps1 -SkipBenchmarks    # Train only, skip evaluation
# ──────────────────────────────────────────────────────────────────────────────

param(
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipDownload,
    [switch]$SkipBenchmarks,
    [switch]$SkipExcel,
    [switch]$EvalOnly
)

$ErrorActionPreference = "Continue"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$WORK = Join-Path $DIR "workdir"
$LOG_DIR = Join-Path $WORK "logs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = Join-Path $LOG_DIR "run_$TIMESTAMP.log"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
if ($EvalOnly) { $SkipData = $true; $SkipTrain = $true; $SkipDownload = $true }

function Log($msg) {
    $ts = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $ts
    Add-Content -Path $LOG_FILE -Value $ts
}

Log "============================================================"
Log "  AnchorGRPO Pipeline"
Log "============================================================"
Log "Work: $WORK"

# Activate venv
$activate = Join-Path $WORK ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) { . $activate; Log "Venv activated" }

# Verify GPU
$gpuCheck = python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>$null
if ($gpuCheck -ne "OK") { Log "CUDA not available!"; exit 1 }
$gpuName = python -c "import torch; props=torch.cuda.get_device_properties(0); mem=getattr(props,'total_mem',getattr(props,'total_memory',0)); print(f'{torch.cuda.get_device_name(0)} ({mem/1024**3:.0f}GB)')"
Log "GPU: $gpuName"

Set-Location $WORK

# ─── Step 1: Generate CRAG Training Data ──────────────────────────────────────
if (-not $SkipData) {
    Log ""
    Log "━━━ STEP 1: Generate CRAG Training Data ━━━"
    python "$DIR\train.py" --step data 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    if ($LASTEXITCODE -ne 0) { Log "FAIL: data gen"; exit 1 }
} else { Log "Skipping data gen" }

# ─── Step 2: Train with GRPO ──────────────────────────────────────────────────
if (-not $SkipTrain) {
    Log ""
    Log "━━━ STEP 2: AnchorGRPO Training (~2-4 hours) ━━━"
    $dataFile = Join-Path $WORK "training_data\training_data.jsonl"
    if (-not (Test-Path $dataFile)) { Log "No training data!"; exit 1 }
    python "$DIR\train.py" --step train 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    if ($LASTEXITCODE -ne 0) { Log "FAIL: training"; exit 1 }

    Log ""
    Log "━━━ STEP 3: Merge LoRA ━━━"
    python "$DIR\train.py" --step merge 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    if ($LASTEXITCODE -ne 0) { Log "FAIL: merge"; exit 1 }
} else { Log "Skipping training" }

# ─── Step 4: Download Datasets ────────────────────────────────────────────────
if (-not $SkipDownload) {
    Log ""
    Log "━━━ STEP 4: Download Datasets (NQ, HotpotQA, MuSiQue) ━━━"
    python "$DIR\download_datasets.py" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
} else { Log "Skipping dataset download" }

# ─── Step 5: Run All Benchmarks ───────────────────────────────────────────────
if (-not $SkipBenchmarks) {
    Log ""
    Log "━━━ STEP 5: Run All Benchmarks ━━━"
    $marg = ""
    if (Test-Path (Join-Path $WORK "merged_model")) { $marg = "--model workdir\merged_model" }
    Invoke-Expression "python `"$DIR\run_all_benchmarks.py`" $marg" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
} else { Log "Skipping benchmarks" }

# ─── Step 6: Export Excel ──────────────────────────────────────────────────────
if (-not $SkipExcel) {
    Log ""
    Log "━━━ STEP 6: Export Excel ━━━"
    python "$DIR\export_excel.py" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
} else { Log "Skipping Excel export" }

# ─── Done ─────────────────────────────────────────────────────────────────────
Log ""
Log "============================================================"
Log "  COMPLETE"
Log "============================================================"
Log "Results: $(Join-Path $WORK 'results')"
Log "Excel:   $(Join-Path $WORK 'results' 'all_benchmarks.xlsx')"
Log ""

$resultsDir = Join-Path $WORK "results"
if (Test-Path $resultsDir) {
    Get-ChildItem -Path $resultsDir -Filter "*.json" | ForEach-Object { Log "  $($_.Name)" }
}
