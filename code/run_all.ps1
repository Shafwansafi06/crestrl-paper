# ──────────────────────────────────────────────────────────────────────────────
# AnchorGRPO Full Pipeline — Windows PowerShell
# Quadro RTX 5000/6000 (16GB VRAM)
# ──────────────────────────────────────────────────────────────────────────────

param(
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipCrag,
    [switch]$EvalOnly,
    [string]$MergedModel = ""
)

$ErrorActionPreference = "Continue"
$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$WORK = Join-Path $DIR "workdir"
$LOG_DIR = Join-Path $WORK "logs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = Join-Path $LOG_DIR "run_$TIMESTAMP.log"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
if ($EvalOnly) { $SkipData = $true; $SkipTrain = $true; $SkipCrag = $true }

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

# ─── Step 1: Data ─────────────────────────────────────────────────────────────
if (-not $SkipData) {
    Log ""
    Log "━━━ STEP 1/5: Data Generation ━━━"
    python "$DIR\train.py" --step data 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    if ($LASTEXITCODE -ne 0) { Log "FAIL: data gen"; exit 1 }
} else { Log "Skipping data gen" }

# ─── Step 2: Train ────────────────────────────────────────────────────────────
if (-not $SkipTrain) {
    Log ""
    Log "━━━ STEP 2/5: AnchorGRPO Training ━━━"
    $dataFile = Join-Path $WORK "training_data"
    $dataFile = Join-Path $dataFile "training_data.jsonl"
    if (-not (Test-Path $dataFile)) { Log "No training data!"; exit 1 }
    python "$DIR\train.py" --step train 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    if ($LASTEXITCODE -ne 0) { Log "FAIL: training"; exit 1 }

    Log "Merging LoRA..."
    python "$DIR\train.py" --step merge 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
    $MergedModel = "merged_model"
} else { Log "Skipping training" }

# ─── Step 3: CRAG ─────────────────────────────────────────────────────────────
if (-not $SkipCrag) {
    Log ""
    Log "━━━ STEP 3/5: CRAG Benchmark ━━━"
    $marg = ""
    if ($MergedModel -and (Test-Path $MergedModel)) { $marg = "--model $MergedModel" }
    Invoke-Expression "python `"$DIR\run_crag_benchmark.py`" --both --max-samples 500 $marg" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
} else { Log "Skipping CRAG" }

# ─── Step 4: Evaluate ─────────────────────────────────────────────────────────
Log ""
Log "━━━ STEP 4/5: Head-to-Head Evaluation ━━━"
$marg = ""
if ($MergedModel -and (Test-Path $MergedModel)) { $marg = "--model $MergedModel" }
Invoke-Expression "python `"$DIR\evaluate.py`" $marg" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append

# ─── Step 5: Paper ────────────────────────────────────────────────────────────
Log ""
Log "━━━ STEP 5/5: Paper Results ━━━"
python "$DIR\paper_results.py" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append

# ─── Done ─────────────────────────────────────────────────────────────────────
Log ""
Log "============================================================"
Log "  COMPLETE"
Log "============================================================"
Log "Results: $(Join-Path $WORK 'results')"
Log "Tables:  $(Join-Path $WORK 'paper' 'tables')"
Log "Figures: $(Join-Path $WORK 'paper' 'figures')"
Log ""

$resultsDir = Join-Path $WORK "results"
if (Test-Path $resultsDir) {
    Get-ChildItem -Path $resultsDir -Filter "*.json" | ForEach-Object { Log "  $($_.Name)" }
}

Log ""
Log "Copy the JSON files from workdir\results\ and paste them back."
pause
