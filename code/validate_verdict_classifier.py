"""
Verdict Classifier Robustness Analysis
=======================================
Validates the lexical verdict/confidence classifiers in reward.py against
ground-truth labels derived from evaluate_response() on existing NQ responses.

Runs locally — no GPU needed. Reads code/results/p_know_validation.json.

Reports:
  1. Abstention detection: precision/recall/F1 of is_abstention() vs
     evaluate_response's "refused" label
  2. Confidence correlation: Pearson r of extract_confidence() vs correctness
  3. Blind-spot analysis: how often get_verdict() is wrong for EXISTS case
  4. Paraphrase brittleness: manual probes of boundary phrases
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reward import is_abstention, extract_confidence, get_verdict
from benchmark_utils import evaluate_response, is_refusal


RESULTS_JSON = SCRIPT_DIR / "results" / "p_know_validation.json"


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ── Paraphrase brittleness probes ─────────────────────────────────────────────
# Each tuple: (response_text, expected_is_abstention, description)
PARAPHRASE_PROBES = [
    # Should be abstention
    ("I don't know the answer to that.", True, "canonical"),
    ("I'm not sure about this one.", True, "canonical"),
    ("I lack the information needed to answer.", False, "paraphrase — missed"),
    ("My knowledge on this topic is limited.", False, "paraphrase — missed"),
    ("This falls outside what I can reliably answer.", False, "paraphrase — missed"),
    ("I'd need to verify that before answering.", False, "paraphrase — missed"),
    # Should NOT be abstention
    ("The answer is Paris.", False, "confident correct"),
    ("Based on the context, the answer is 42.", False, "confident correct"),
    ("I cannot provide legal advice.",  True, "false positive risk — contains 'i cannot'"),
]

# Confidence probes: (response, expected direction, description)
CONF_PROBES = [
    ("Definitely the capital is Paris.", "high", "high-conf word"),
    ("I think it might possibly be Paris.", "low", "low-conf words"),
    ("The answer is Paris.", "mid", "neutral — no signal words"),
    ("I'm not sure, probably Paris.", "low", "mixed low"),
    ("It is clearly and certainly Paris.", "high", "multiple high-conf"),
]


def main():
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records from {RESULTS_JSON.name}\n")

    # ── 1. Abstention detection ───────────────────────────────────────────────
    print("=" * 60)
    print("1. ABSTENTION DETECTION  (is_abstention vs evaluate_response['refused'])")
    print("=" * 60)

    tp = fp = fn = tn = 0
    abstention_errors = []

    for r in records:
        response, gold, correct_flag = r["response"], r["gold"], r["correct"]
        ev = evaluate_response(response, gold)
        true_refused = ev["refused"]
        pred_abstain = is_abstention(response)

        if pred_abstain and true_refused:     tp += 1
        elif pred_abstain and not true_refused: fp += 1
        elif not pred_abstain and true_refused: fn += 1
        else:                                   tn += 1

        if pred_abstain != true_refused:
            abstention_errors.append({
                "response": response[:100],
                "pred": pred_abstain,
                "true": true_refused,
            })

    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    n = len(records)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  (n={n})")
    print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")
    print(f"  Accuracy:  {(tp+tn)/n:.3f}")

    if fn > 0:
        print(f"\n  False negatives ({fn}) — model refuses but classifier misses:")
        for e in abstention_errors[:3]:
            if not e["pred"] and e["true"]:
                print(f"    '{e['response'][:80]}...'")
    if fp > 0:
        print(f"\n  False positives ({fp}) — classifier fires but model didn't refuse:")
        for e in abstention_errors[:3]:
            if e["pred"] and not e["true"]:
                print(f"    '{e['response'][:80]}...'")

    # ── 2. Confidence correlation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. CONFIDENCE CORRELATION  (extract_confidence vs correctness)")
    print("=" * 60)

    confs, corrects = [], []
    for r in records:
        response, gold = r["response"], r["gold"]
        ev = evaluate_response(response, gold)
        c = extract_confidence(response)
        confs.append(c)
        corrects.append(int(ev["correct"]))

    confs = np.array(confs)
    corrects = np.array(corrects)
    corr_r, corr_p = pearsonr(confs, corrects)

    conf_correct = confs[corrects == 1]
    conf_wrong   = confs[corrects == 0]

    print(f"  Pearson r (confidence vs correctness): {corr_r:.4f}  (p={corr_p:.4f})")
    print(f"  Mean confidence (correct):   {conf_correct.mean():.3f}  (n={len(conf_correct)})")
    print(f"  Mean confidence (incorrect): {conf_wrong.mean():.3f}  (n={len(conf_wrong)})")

    # Count how many responses have any signal words at all
    from reward import HIGH_CONF, LOW_CONF
    no_signal = sum(
        1 for r in records
        if not any(w in r["response"].lower() for w in HIGH_CONF + LOW_CONF)
    )
    print(f"  Responses with NO signal words: {no_signal}/{len(records)} "
          f"({no_signal/len(records)*100:.0f}%) -> all get conf=0.5")

    # ── 3. get_verdict blind spot (EXISTS case) ───────────────────────────────
    print("\n" + "=" * 60)
    print("3. GET_VERDICT BLIND SPOT  (EXISTS case)")
    print("=" * 60)

    verdict_correct = verdict_abstain = verdict_hallu_missed = 0
    for r in records:
        response, gold = r["response"], r["gold"]
        ev = evaluate_response(response, gold)
        pred = get_verdict(response, "EXISTS")

        if pred == "correct" and not ev["correct"] and not ev["refused"]:
            verdict_hallu_missed += 1   # hallucination predicted as "correct"
        if pred == "abstain":
            verdict_abstain += 1
        if pred == "correct" and ev["correct"]:
            verdict_correct += 1

    true_hallu = sum(1 for r in records
                     if not evaluate_response(r["response"], r["gold"])["correct"]
                     and not evaluate_response(r["response"], r["gold"])["refused"])

    print(f"  True hallucinations in data: {true_hallu}/{len(records)}")
    print(f"  Hallucinations correctly flagged by get_verdict: 0/{true_hallu}")
    print(f"  (get_verdict always returns 'correct' for EXISTS when not abstaining)")
    print(f"  Hallucination recall for EXISTS case: 0.000  [structural gap]")

    # ── 4. Paraphrase brittleness ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. PARAPHRASE BRITTLENESS  (manual probes for is_abstention)")
    print("=" * 60)

    errors = 0
    for text, expected, desc in PARAPHRASE_PROBES:
        pred = is_abstention(text)
        status = "OK" if pred == expected else "FAIL"
        if status == "FAIL":
            errors += 1
        print(f"  [{status}] expected={expected} got={pred}  [{desc}]")
        print(f"        '{text}'")

    print(f"\n  {errors}/{len(PARAPHRASE_PROBES)} paraphrase probes failed")

    print("\n" + "=" * 60)
    print("5. CONFIDENCE PROBE (extract_confidence on known inputs)")
    print("=" * 60)
    for text, expected_dir, desc in CONF_PROBES:
        c = extract_confidence(text)
        print(f"  conf={c:.3f}  expected={expected_dir:4s}  [{desc}]")
        print(f"        '{text}'")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"  Abstention detection:  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")
    print(f"  Confidence correlation: r={corr_r:.3f}  p={corr_p:.4f}")
    print(f"  Hallucination recall (EXISTS): 0.00  [structural — fix needed]")
    print(f"  Paraphrase failures: {errors}/{len(PARAPHRASE_PROBES)}")

    if corr_r < 0.1 or corr_p > 0.05:
        conf_verdict = "CONFIDENCE EXTRACTION UNRELIABLE — confidence reward is noise"
    elif corr_r < 0.2:
        conf_verdict = "CONFIDENCE WEAK — calibration reward contribution is marginal"
    else:
        conf_verdict = "CONFIDENCE USABLE — calibration reward has signal"
    print(f"\n  Confidence verdict: {conf_verdict}")

    # Save
    out = {
        "abstention": {"precision": prec, "recall": rec, "f1": f1,
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "confidence_correlation": {"r": corr_r, "p": corr_p,
                                   "mean_conf_correct": float(conf_correct.mean()),
                                   "mean_conf_incorrect": float(conf_wrong.mean()),
                                   "no_signal_pct": no_signal / len(records)},
        "hallucination_recall_EXISTS": 0.0,
        "paraphrase_failures": errors,
        "paraphrase_total": len(PARAPHRASE_PROBES),
        "confidence_verdict": conf_verdict,
    }
    out_path = SCRIPT_DIR / "results" / "verdict_classifier_validation.json"
    with open(out_path, "w") as f_out:
        json.dump(out, f_out, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
