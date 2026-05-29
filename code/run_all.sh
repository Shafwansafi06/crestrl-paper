#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AnchorGRPO Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$DIR/workdir"
LOG="$WORK/logs/$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$WORK/logs"

SKIP_DATA=false; SKIP_TRAIN=false; SKIP_CRAG=false; MERGED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)    SKIP_DATA=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --skip-crag)    SKIP_CRAG=true; shift ;;
        --eval-only)    SKIP_DATA=true; SKIP_TRAIN=true; SKIP_CRAG=true; shift ;;
        --merged-model) MERGED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "============================================================"
log "  AnchorGRPO Pipeline — RTX 4000 Ada (16GB)"
log "============================================================"

[ -d "$WORK/.venv" ] && source "$WORK/.venv/bin/activate"
python3 -c "import torch; assert torch.cuda.is_available()" || { log "No CUDA"; exit 1; }
log "GPU: $(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB)')")"

cd "$WORK"

# Step 1: Data generation with logit-based p_know
if [ "$SKIP_DATA" = false ]; then
    log ""
    log "━━━ STEP 1/5: Data Generation (logit-based p_know) ━━━"
    python3 "$DIR/train.py" --step data 2>&1 | tee -a "$LOG"
fi

# Step 2: AnchorGRPO training
if [ "$SKIP_TRAIN" = false ]; then
    log ""
    log "━━━ STEP 2/5: AnchorGRPO Training ━━━"
    python3 "$DIR/train.py" --step train 2>&1 | tee -a "$LOG"
    log "Merging LoRA..."
    python3 "$DIR/train.py" --step merge 2>&1 | tee -a "$LOG"
    MERGED="merged_model"
fi

# Step 3: CRAG benchmark
if [ "$SKIP_CRAG" = false ]; then
    log ""
    log "━━━ STEP 3/5: CRAG Benchmark ━━━"
    MARG=""; [ -n "$MERGED" ] && [ -d "$MERGED" ] && MARG="--model $MERGED"
    python3 "$DIR/run_crag_benchmark.py" --both $MARG 2>&1 | tee -a "$LOG"
fi

# Step 4: TruthRL comparison
log ""
log "━━━ STEP 4/5: Head-to-Head Evaluation ━━━"
MARG=""; [ -n "$MERGED" ] && [ -d "$MERGED" ] && MARG="--model $MERGED"
python3 "$DIR/evaluate.py" $MARG 2>&1 | tee -a "$LOG"

# Step 5: Paper results
log ""
log "━━━ STEP 5/5: Paper Results ━━━"
python3 "$DIR/paper_results.py" 2>&1 | tee -a "$LOG"

log ""
log "============================================================"
log "  COMPLETE"
log "============================================================"
log "Results: $WORK/results/"
log "Tables:  $WORK/paper/tables/"
log "Figures: $WORK/paper/figures/"
ls -la results/*.json 2>/dev/null | tee -a "$LOG"
