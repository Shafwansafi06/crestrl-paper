# CrestRL Paper — Progress Log

Track of all changes, problems, and fixes as we work toward ICLR submission.

---

## Phase 0 — Validate the Foundation

### Step 1: p_know Proxy Validation
**File:** `code/validate_p_know.py`
**Status:** Complete. Results in `code/results/p_know_validation.json`

**Problem (original script):**
- Only ~24 samples (too few for reliable correlation)
- Correctness criterion was circular: "correct = didn't refuse" for EXISTS items
- Measured PPL of the model's own *generated* response (model always assigns itself low PPL)
- Unrelated to whether the model actually knows the answer

**Fix:**
- Measure PPL of the **gold answer** given the question — low PPL means the model assigns high probability to the correct answer text, which is the actual p_know claim
- Use `evaluate_response(response, gold)` with true gold answer for correctness
- Sample 300 NQ questions (got 91 with valid short answers after strict filtering)
- Reused `load_nq()` from `run_nq_benchmark.py` instead of re-implementing NQ schema parsing
- Added 95% bootstrap CI on Pearson r
- Auto-verdict output tells you Path A vs Path B

**Issues during implementation:**
- Local disk full — NQ download failed with `os error 112` (no space). Not a code bug; script runs fine on remote machine where data is cached.
- NQ only yielded 91 samples (not 300) because NQ's short-answer field is sparsely populated; most rows have empty `short_answers`. 89 valid after PPL<10k filter. Still sufficient (p=0.001).

**Results on Mistral-7B / NQ validation set:**
| Metric | Value |
|--------|-------|
| Pearson r (1/PPL vs correct) | 0.3416 |
| p-value | 0.0011 |
| 95% CI | [0.214, 0.460] |
| Mean PPL (correct) | 190.4 |
| Mean PPL (incorrect) | 580.3 |
| Verdict | PROXY VALID |

**Paper implication:** Replace the borrowed r=0.73 from Kadavath et al. with our measured r=0.34 on Mistral-7B. Report as "moderate correlation". Lead with the 3x PPL gap (190 vs 580) as the cleaner signal. Do not claim r=0.73.

---

## Phase 1 — Cheap Experiments (No Retraining)

### Step 2: Verdict Classifier Robustness
**File:** `code/validate_verdict_classifier.py`
**Status:** Complete. Results in `code/results/verdict_classifier_validation.json`

**What was tested:**
- `is_abstention()`: precision/recall vs `evaluate_response()["refused"]` on 91 NQ responses
- `extract_confidence()`: Pearson r vs actual correctness
- `get_verdict()` blind spot for EXISTS case (all NQ questions have real answers)
- Paraphrase brittleness probes for `is_abstention()`

**Issues during implementation:**
- `UnicodeEncodeError` on Windows (cp1252 codec): `→` arrow character in print statement. Fixed by replacing `→` with `->`.
- Variable name conflict: `r` used both as `precision_recall_f1()` return value and as loop variable `for r in records`. Fixed by renaming to `prec, rec, f1`.

**Results:**
| Metric | Value | Implication |
|--------|-------|-------------|
| Abstention P/R/F1 | 1.00 / 1.00 / 1.00 | is_abstention() works perfectly on NQ |
| Confidence r vs correct | 0.049 (p=0.65) | extract_confidence() is noise |
| Responses with no signal words | 47% | Defaulting to conf=0.5 half the time |
| Hallucination recall (EXISTS) | 0.00 | Structural gap: get_verdict() can't detect hallucinations for EXISTS case |
| Paraphrase failures | 0/9 | is_abstention() robust to tested paraphrases |

**Critical finding — `extract_confidence()` bug:**
`HIGH_CONF` list contains common verbs ("is a", "are", "will", "it is") that fire on almost every factual answer, not just high-confidence ones. This is why conf=0.9 for "The answer is Paris" (a neutral response). The phrase-based approach has no real signal.

**Critical finding — `get_verdict()` structural gap:**
`is_hallucination(r, expected)` only executes for `expected == "NOT_EXISTS"`. For all EXISTS-case questions (NQ, HotpotQA, real-world RAG), hallucination is structurally undetectable by the current classifier. 26/91 true hallucinations in the data were all classified as "correct".

**Paper implication:**
- The `w_c=0.16` calibration reward was based on noisy confidence scores. Fix required before claiming the calibration component contributes anything.
- The classifier is valid only for the adversarial benchmark (NOT_EXISTS items). On real-world datasets, `evaluate_response()` (token overlap) handles correctness detection, not `get_verdict()`. The paper must clearly separate these two evaluation paths.

### Step 2 Fix: extract_confidence replaced with p_know
**File:** `code/reward.py` — `compute_live_reward()`
**Status:** Complete.

**Change:** Replaced `conf = extract_confidence(completion)` with `conf = p_know`.

**Rationale:**
- `p_know` (mean exp(log_prob) of generated tokens) is already computed in `compute_live_reward`
- It's the same mechanism validated in Step 1 (r=0.34, p=0.001)
- `extract_confidence()` had r=0.049 (not significant) and 47% no-signal rate
- `extract_confidence()` function retained in reward.py for post-hoc text-only analysis

---

## Open Issues (Remaining Plan)

### Phase 1 (remaining)
- [ ] **Step 3:** Reward-component ablation table (re-score existing generations under outcome-only / +calibration / +anchor / full)
- [ ] **Step 4:** Variance + CIs on all single-run numbers (inference times, per-category accuracies need ≥5 seeds)
- [ ] **Step 5:** Sensitivity sweep on gamma (1.5, 2, 3) and weights (±20%) at reward-scoring level
- [ ] **Step 6:** MuSiQue diagnosis — inspect 20 transcripts; fix prompt or drop dataset

### Phase 2 (requires GPU on remote machine)
- [ ] Switch BASE_MODEL to Qwen2.5-1.5B, set num_generations=32, train ≥1000 steps
- [ ] Expand training set past 198 prompts
- [ ] Controlled TruthRL baseline (same model/data/steps as CrestRL run)

### Phase 3 (writing)
- [ ] Promote T_b metric from footnote to primary contribution
- [ ] Replace Kadavath r=0.73 citation with measured r=0.34
- [ ] Report confidence component fix (phrase-based → p_know)
- [ ] Scope claims honestly: state validated operating regime

---

## Key Numbers for the Paper

| Claim | Source | Value |
|-------|--------|-------|
| p_know proxy correlation on Mistral-7B/NQ | validate_p_know.py | r=0.34, p=0.001, CI[0.21,0.46] |
| PPL gap correct vs incorrect | validate_p_know.py | 190 vs 580 (3x) |
| Abstention detection F1 | validate_verdict_classifier.py | 1.00 |
| Confidence extraction correlation | validate_verdict_classifier.py | r=0.049 (not significant) |
| Hallucination recall EXISTS case | validate_verdict_classifier.py | 0.00 (structural) |
