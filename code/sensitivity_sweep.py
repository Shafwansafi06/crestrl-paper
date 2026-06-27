"""
Sensitivity Sweep — gamma and reward weights
=============================================
Reads existing ablation_reward.json records (verdict + p_know + gt_correct)
and re-scores under different gamma / weight configs. Pure Python, no GPU.

Answers: Is gamma=2 actually optimal? Are the weights stable to ±20% perturbation?

Run locally:
    python sensitivity_sweep.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from itertools import product
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

RECORDS_PATH = SCRIPT_DIR / "results" / "ablation_reward.json"

# ── Reward functions (inline — no model needed) ───────────────────────────────

def outcome_reward(verdict, p_know, delta=0.1):
    p = float(np.clip(p_know, 0, 1))
    if verdict == "correct":      return 1.0
    if verdict == "abstain":      return (0.5 - p) if delta < p < (1 - delta) else 0.0
    return -(1.0 + p)  # hallucination

def calibration_reward(verdict, p_know, lam, gamma):
    c = float(np.clip(p_know, 0, 1))
    if verdict == "correct":      return lam * c
    if verdict == "hallucination": return -gamma * lam * c
    return lam * (1.0 - c)  # abstain

def anchor_reward(verdict, p_know):
    if verdict == "hallucination": return -p_know
    if verdict == "correct":       return 1.0 - p_know
    return 0.0

def full_reward(verdict, p_know, w_o, w_c, lam_a, lam_c, gamma, eps=0.05):
    r_out = outcome_reward(verdict, p_know)
    r_cal = calibration_reward(verdict, p_know, lam_c, gamma)
    r_anc = anchor_reward(verdict, p_know)
    total = w_o * r_out + lam_a * r_anc + w_c * r_cal + eps
    return float(np.clip(total, -2.5, 1.5))


def sweep_gamma(records, gammas=(1.5, 2.0, 2.5, 3.0)):
    """Fix weights at baseline, sweep gamma."""
    gt = np.array([r["gt_correct"] for r in records], dtype=float)

    print("\n" + "=" * 60)
    print("GAMMA SWEEP  (w_o=0.83, lam_a=0.40, w_c=0.16 fixed)")
    print("=" * 60)
    print(f"  {'gamma':>6}  {'r (all)':>10}  {'r (nq)':>10}  {'r (adv)':>10}  {'p-value':>10}")
    print(f"  {'-'*56}")

    nq_mask  = np.array([r["source"] == "nq" for r in records])
    adv_mask = ~nq_mask
    gt_nq    = gt[nq_mask]
    gt_adv   = gt[adv_mask]

    results = []
    for gamma in gammas:
        rewards = np.array([
            full_reward(r["verdict"], r["p_know"],
                        w_o=0.83, w_c=0.16, lam_a=0.40, lam_c=0.22, gamma=gamma)
            for r in records
        ])
        r_all, p_all = pearsonr(rewards, gt)
        r_nq,  _     = pearsonr(rewards[nq_mask],  gt_nq)
        r_adv, _     = pearsonr(rewards[adv_mask], gt_adv)

        marker = "  <-- baseline" if gamma == 2.0 else ""
        print(f"  {gamma:>6.1f}  {r_all:>10.4f}  {r_nq:>10.4f}  {r_adv:>10.4f}  {p_all:>10.4f}{marker}")
        results.append({"gamma": gamma, "r_all": round(r_all, 4), "r_nq": round(r_nq, 4),
                         "r_adv": round(r_adv, 4), "p_all": round(p_all, 6)})
    return results


def sweep_weights(records):
    """Perturb each weight ±20%, one at a time, fixing the others."""
    gt = np.array([r["gt_correct"] for r in records], dtype=float)
    baseline = {"w_o": 0.83, "lam_a": 0.40, "w_c": 0.16, "lam_c": 0.22, "gamma": 2.0}

    baseline_r, _ = pearsonr(
        np.array([full_reward(r["verdict"], r["p_know"], **baseline) for r in records]), gt
    )

    print("\n" + "=" * 60)
    print("WEIGHT PERTURBATION  (±20%, one at a time, gamma=2.0)")
    print("=" * 60)
    print(f"  {'param':<8}  {'value':>8}  {'r (all)':>10}  {'delta_r':>10}  {'change':>8}")
    print(f"  {'-'*56}")
    print(f"  {'baseline':<8}  {'':>8}  {baseline_r:>10.4f}  {'':>10}  {'':>8}")

    param_ranges = {
        "w_o":   [round(0.83 * f, 3) for f in (0.8, 1.0, 1.2)],
        "lam_a": [round(0.40 * f, 3) for f in (0.8, 1.0, 1.2)],
        "w_c":   [round(0.16 * f, 3) for f in (0.8, 1.0, 1.2)],
        "lam_c": [round(0.22 * f, 3) for f in (0.8, 1.0, 1.2)],
    }

    results = []
    for param, values in param_ranges.items():
        for val in values:
            if val == baseline[param]:
                continue
            cfg = {**baseline, param: val}
            rewards = np.array([
                full_reward(r["verdict"], r["p_know"], **cfg) for r in records
            ])
            r_val, _ = pearsonr(rewards, gt)
            delta = r_val - baseline_r
            direction = "stable" if abs(delta) < 0.005 else ("better" if delta > 0 else "worse")
            marker = " <-- baseline" if val == baseline[param] else ""
            print(f"  {param:<8}  {val:>8.3f}  {r_val:>10.4f}  {delta:>+10.4f}  {direction:>8}{marker}")
            results.append({"param": param, "value": val, "r_all": round(r_val, 4),
                             "delta_r": round(delta, 4), "direction": direction})

    return results, round(baseline_r, 4)


def main():
    with open(RECORDS_PATH) as f:
        data = json.load(f)
    records = data["records"]
    print(f"Loaded {len(records)} records  "
          f"(nq={sum(1 for r in records if r['source']=='nq')}, "
          f"adv={sum(1 for r in records if r['source']=='adversarial')})")

    gamma_results  = sweep_gamma(records)
    weight_results, baseline_r = sweep_weights(records)

    # ── Verdict-conditional analysis ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REWARD DISTRIBUTION BY VERDICT  (gamma=2, baseline weights)")
    print("=" * 60)
    for v in ("correct", "abstain", "hallucination"):
        subset = [r for r in records if r["verdict"] == v]
        if not subset:
            continue
        rewards = [full_reward(r["verdict"], r["p_know"],
                               w_o=0.83, w_c=0.16, lam_a=0.40, lam_c=0.22, gamma=2.0)
                   for r in subset]
        print(f"  {v:<15}  n={len(subset):>3}  "
              f"mean={np.mean(rewards):>+7.4f}  std={np.std(rewards):>6.4f}  "
              f"min={np.min(rewards):>+7.4f}  max={np.max(rewards):>+7.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    best_gamma = max(gamma_results, key=lambda x: x["r_all"])
    unstable   = [r for r in weight_results if r["direction"] != "stable"]

    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"  Baseline r (gamma=2.0):       {baseline_r:.4f}")
    print(f"  Best gamma:                   {best_gamma['gamma']} (r={best_gamma['r_all']:.4f})")
    print(f"  Gamma range tested:           1.5 – 3.0")
    print(f"  Weight perturbations unstable: {len(unstable)}/{len(weight_results)}")
    if unstable:
        for u in unstable:
            print(f"    {u['param']}={u['value']} -> delta_r={u['delta_r']:+.4f} ({u['direction']})")
    else:
        print("    None — all weight perturbations are stable (|delta_r| < 0.005)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "baseline_r": baseline_r,
        "gamma_sweep": gamma_results,
        "weight_perturbation": weight_results,
        "best_gamma": best_gamma["gamma"],
        "n_unstable_weights": len(unstable),
    }
    out_path = SCRIPT_DIR / "results" / "sensitivity_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
