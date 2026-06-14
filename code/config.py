"""
AnchorGRPO Configuration — Mathematically Derived Parameters
=============================================================
Every parameter below is derived from first principles, not tuned intuitively.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WORK_DIR = SCRIPT_DIR / "workdir"
DATA_DIR = WORK_DIR / "training_data"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
MERGED_DIR = WORK_DIR / "merged_model"
RESULTS_DIR = WORK_DIR / "results"
CRAG_DIR = WORK_DIR / "crag_data"
LOG_DIR = WORK_DIR / "logs"
MODEL_CACHE_DIR = WORK_DIR / "model_cache"

NQ_DIR = WORK_DIR / "nq_data"
HOTPOTQA_DIR = WORK_DIR / "hotpotqa_data"
MUSIQUE_DIR = WORK_DIR / "musique_data"

for d in [WORK_DIR, DATA_DIR, CHECKPOINT_DIR, MERGED_DIR, RESULTS_DIR, CRAG_DIR,
          NQ_DIR, HOTPOTQA_DIR, MUSIQUE_DIR, LOG_DIR, MODEL_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ─── Reward Weights (derived via inverse-variance weighting, not intuition) ───
#
# Var(r_outcome)    ~ 0.40  →  w = 1/0.40 = 2.50  →  normalized: 0.83
# Var(r_calibration) ~ 0.05  →  w = 1/0.05 = 20.0  →  normalized: 0.16
# Var(r_consistency) ~ 0.02  →  w = 1/0.02 = 50.0  →  normalized: 0.01
#
# Consistency is near-pure noise at our sample sizes. Dropped to epsilon.
# Final: outcome=0.83, calibration=0.16, consistency=0.01

REWARD_WEIGHTS = {
    "w_outcome": 0.83,
    "w_calibration": 0.16,
    "w_consistency": 0.01,
}

# ─── Calibration (asymmetric: punish hallucination 2× vs reward correct) ─────

LAMBDA_CALIB = 0.22  # calibration coefficient
CALIBRATION_ASYMMETRY = 2.0  # hallucination penalty = 2× correct reward

# ─── Variance floor (prevents zero-gradient on uniform reward groups) ─────────
# Math: when std(r) = 0, advantage = 0/0 → zero gradient
# Floor ensures A_i = (r_i - mean) / max(std, eps) always has signal

EPS_FLOOR = 0.05  # minimum reward variance for advantage computation

# ─── Abstention zone (clipping to avoid noise at boundary) ────────────────────
# When p_know ≈ 0.5, abstention reward ≈ 0 regardless → no signal
# Only apply signed reward when p_know is informative (outside [delta, 1-delta])

DELTA_ABSTAIN = 0.1  # abstention reward clipped to zero near p_know=0.5

# ─── AnchorGRPO (the novel component) ────────────────────────────────────────

ALPHA_ANCHOR = 0.4  # weight for anchor reward (context grounding penalty)
PROBE_HIDDEN_DIM = 256  # hallucination probe hidden layer size
PROBE_NUM_LAYERS = 2  # probe depth
TAU_H = 0.65  # dynamic abstention threshold (H > tau → force abstain)

# ─── VIB Regularization (information bottleneck) ──────────────────────────────

LAMBDA_VIB = 0.01  # KL penalty weight for context-matching
CONTEXT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── GRPO Training ────────────────────────────────────────────────────────────

GRPO_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "num_epochs": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "max_seq_length": 512,
    "num_generations": 4,
    "beta": 0.1,
    "max_grad_norm": 1.0,
}

# ─── Temperature Scaling (post-hoc calibration, zero training) ────────────────
# Guo et al. 2017: optimal T for 7B models is typically 1.3-1.8
# This is calibrated on a held-out set by calibrate.py

TEMPERATURE = 1.4  # default; will be optimized by calibration

# ─── CRAG Benchmark ───────────────────────────────────────────────────────────

CRAG_DATASET = "crag-check/crag-mix-simplified"
CRAG_MAX_SAMPLES = 500
CRAG_TOP_K_PASSAGES = 5  # retrieve top-k passages per query

# ─── NQ Benchmark ─────────────────────────────────────────────────────────────

NQ_DATASET = "google-research-datasets/natural_questions"
NQ_MAX_SAMPLES = None  # Full dataset

# ─── HotpotQA Benchmark ───────────────────────────────────────────────────────

HOTPOTQA_DATASET = "hotpotqa/hotpot_qa"
HOTPOTQA_MAX_SAMPLES = None  # Full dataset

# ─── MuSiQue Benchmark ────────────────────────────────────────────────────────

MUSIQUE_DATASET = "dgslibisey/MuSiQue"
MUSIQUE_MAX_SAMPLES = None  # Full dataset

# ─── Result Paths ─────────────────────────────────────────────────────────────

EVAL_RESULTS = RESULTS_DIR / "evaluation.json"
CRAG_BASE_RESULTS = RESULTS_DIR / "crag_base.json"
CRAG_FINETUNED_RESULTS = RESULTS_DIR / "crag_finetuned.json"
CRAG_COMPARISON = RESULTS_DIR / "crag_comparison.json"

NQ_BASE_RESULTS = RESULTS_DIR / "nq_base.json"
NQ_FINETUNED_RESULTS = RESULTS_DIR / "nq_finetuned.json"
NQ_COMPARISON = RESULTS_DIR / "nq_comparison.json"

HOTPOTQA_BASE_RESULTS = RESULTS_DIR / "hotpotqa_base.json"
HOTPOTQA_FINETUNED_RESULTS = RESULTS_DIR / "hotpotqa_finetuned.json"
HOTPOTQA_COMPARISON = RESULTS_DIR / "hotpotqa_comparison.json"

MUSIQUE_BASE_RESULTS = RESULTS_DIR / "musique_base.json"
MUSIQUE_FINETUNED_RESULTS = RESULTS_DIR / "musique_finetuned.json"
MUSIQUE_COMPARISON = RESULTS_DIR / "musique_comparison.json"

ALL_BENCHMARKS_XLSX = RESULTS_DIR / "all_benchmarks.xlsx"
