"""
MuSiQue Diagnosis
==================
Explains the 95% refusal rate without needing saved model responses.

Root hypothesis: BM25 retrieval systematically fails on multi-hop questions
because the relevant paragraphs for hop-2/hop-3 score low on keyword overlap
with the question. The GROUNDED_PROMPT then correctly instructs the model to
say "I don't know" when context is insufficient.

This script measures retrieval quality directly (no model needed) on 100
MuSiQue samples, then generates 20 actual responses to confirm the pattern.

Run on remote machine:
    python diagnose_musique.py
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from run_musique_benchmark import load_musique
from benchmark_utils import retrieve_passages_bm25, is_refusal, GROUNDED_PROMPT, NO_CONTEXT_PROMPT


def answer_in_context(answer: str, context_passages: list[str]) -> bool:
    """Check if the gold answer string appears in any retrieved passage."""
    ans = answer.lower().strip()
    combined = " ".join(p.lower() for p in context_passages)
    return ans in combined


def answer_in_full_context(answer: str, all_passages: list[str]) -> bool:
    """Check if gold answer appears anywhere in the full paragraph set."""
    ans = answer.lower().strip()
    combined = " ".join(p.lower() for p in all_passages)
    return ans in combined


def retrieval_quality(samples: list[dict], top_k: int = 5) -> dict:
    """
    For each sample, check if BM25 top-k retrieved passages contain the gold answer.
    Also check if the full paragraph set contains it (to detect unanswerable items).
    """
    hits_retrieved = 0
    hits_full = 0
    n_no_context = 0
    n_total = len(samples)

    passage_counts = []
    retrieval_details = []

    for s in samples:
        all_passages = s.get("context", [])
        if isinstance(all_passages, str):
            all_passages = [all_passages] if all_passages else []

        passage_counts.append(len(all_passages))

        if not all_passages:
            n_no_context += 1
            retrieval_details.append({
                "question": s["question"][:80],
                "answer": s["answer"],
                "n_passages": 0,
                "answer_in_retrieved": False,
                "answer_in_full": False,
            })
            continue

        retrieved = retrieve_passages_bm25(s["question"], all_passages, top_k=top_k)
        in_retrieved = answer_in_context(s["answer"], retrieved)
        in_full = answer_in_full_context(s["answer"], all_passages)

        if in_retrieved:
            hits_retrieved += 1
        if in_full:
            hits_full += 1

        retrieval_details.append({
            "question": s["question"][:80],
            "answer": s["answer"],
            "n_passages": len(all_passages),
            "n_retrieved": len(retrieved),
            "answer_in_retrieved": in_retrieved,
            "answer_in_full": in_full,
        })

    return {
        "n_total": n_total,
        "n_no_context": n_no_context,
        "retrieval_recall": hits_retrieved / n_total,
        "full_context_recall": hits_full / n_total,
        "mean_passage_count": float(np.mean(passage_counts)) if passage_counts else 0,
        "details": retrieval_details,
    }


def generate_sample_responses(n: int = 20) -> list[dict]:
    """Generate n responses with the model to confirm the refusal pattern."""
    import torch
    from benchmark_utils import load_model

    print(f"\nLoading model for {n} sample responses...")
    model, tokenizer = load_model()

    samples = load_musique(max_samples=n * 5)[:n]
    records = []

    for i, s in enumerate(samples):
        all_passages = s.get("context", [])
        if isinstance(all_passages, str):
            all_passages = [all_passages] if all_passages else []

        retrieved = retrieve_passages_bm25(s["question"], all_passages, top_k=5)
        answer_recoverable = answer_in_context(s["answer"], retrieved)

        if retrieved:
            context_str = "\n\n".join(
                f"[Source {j+1}]: {p}" for j, p in enumerate(retrieved)
            )
            prompt = GROUNDED_PROMPT.format(context=context_str, query=s["question"])
        else:
            prompt = NO_CONTEXT_PROMPT.format(query=s["question"])

        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=150, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        refused = is_refusal(response)
        records.append({
            "question": s["question"],
            "gold": s["answer"],
            "n_passages_total": len(all_passages),
            "answer_in_retrieved": answer_recoverable,
            "response": response[:200],
            "refused": refused,
        })
        torch.cuda.empty_cache()
        print(f"  [{i+1}/{n}] refused={refused} recoverable={answer_recoverable}  "
              f"q={s['question'][:50]}...")

    return records


def main():
    print("=" * 60)
    print("MuSiQue Diagnosis")
    print("=" * 60)

    # ── Phase 1: Retrieval quality (no model needed) ──────────────────────────
    print("\nLoading 200 MuSiQue samples for retrieval analysis...")
    samples = load_musique(max_samples=200)
    print(f"Loaded {len(samples)} samples")

    rq = retrieval_quality(samples, top_k=5)

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY ANALYSIS  (BM25 top-5)")
    print("=" * 60)
    print(f"  Samples analysed:              {rq['n_total']}")
    print(f"  Mean paragraphs per question:  {rq['mean_passage_count']:.1f}")
    print(f"  Samples with no context:       {rq['n_no_context']}")
    print(f"  Answer in retrieved top-5:     {rq['retrieval_recall']*100:.1f}%")
    print(f"  Answer in FULL paragraph set:  {rq['full_context_recall']*100:.1f}%")
    print()
    print(f"  BM25 retrieval gap:  {(rq['full_context_recall'] - rq['retrieval_recall'])*100:.1f}pp")
    print(f"  (paragraphs exist but BM25 doesn't surface them)")

    # Show 5 failure examples
    failures = [d for d in rq["details"]
                if d["answer_in_full"] and not d["answer_in_retrieved"]][:5]
    if failures:
        print(f"\n  Retrieval failures (answer exists but not retrieved):")
        for f in failures:
            print(f"    Q: {f['question']}")
            print(f"    A: {f['answer']}  | passages={f['n_passages']}")

    # ── Phase 2: Prompt / abstain phrase analysis (no model needed) ───────────
    print("\n" + "=" * 60)
    print("ABSTAIN PHRASE COVERAGE ANALYSIS")
    print("=" * 60)
    from benchmark_utils import ABSTAIN_PHRASES
    print(f"  ABSTAIN_PHRASES has {len(ABSTAIN_PHRASES)} triggers:")
    broad = [p for p in ABSTAIN_PHRASES if len(p.split()) <= 2]
    print(f"  Short (<=2 words, high false-positive risk): {broad}")
    print()
    print("  GROUNDED_PROMPT explicitly says:")
    print('  "If the sources don\'t contain enough information, say I don\'t know"')
    print("  -> With BM25 failing to retrieve relevant paragraphs, model is")
    print("     CORRECT to refuse. 95% refusal = 95% retrieval failure rate.")

    # ── Phase 3: 20 live responses to confirm ─────────────────────────────────
    print("\n" + "=" * 60)
    print("LIVE RESPONSE SAMPLE  (n=20)")
    print("=" * 60)
    live = generate_sample_responses(n=20)

    refused_total = sum(1 for r in live if r["refused"])
    refused_unrecoverable = sum(
        1 for r in live if r["refused"] and not r["answer_in_retrieved"]
    )
    refused_recoverable = sum(
        1 for r in live if r["refused"] and r["answer_in_retrieved"]
    )

    print(f"\n  Refused:                  {refused_total}/20  ({refused_total*5}%)")
    print(f"  Refused & unrecoverable:  {refused_unrecoverable}/20  (BM25 failed -> expected)")
    print(f"  Refused & recoverable:    {refused_recoverable}/20  (model over-refuses)")

    print("\n  Sample transcripts:")
    for r in live[:5]:
        status = "REFUSED" if r["refused"] else "ANSWERED"
        recov  = "recoverable" if r["answer_in_retrieved"] else "BM25-failed"
        print(f"\n  [{status}] [{recov}]")
        print(f"  Q: {r['question']}")
        print(f"  Gold: {r['gold']}")
        print(f"  Response: {r['response'][:150]}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    retrieval_recall_pct = rq["retrieval_recall"] * 100

    if retrieval_recall_pct < 30:
        verdict = (
            "DROP MuSiQue. BM25 retrieval recall is too low for multi-hop questions. "
            "The 95% refusal rate is a retrieval artefact, not a model behaviour. "
            "Including it in the paper confounds all comparisons."
        )
    else:
        verdict = (
            "FIX retrieval. Replace BM25 top-5 with all paragraphs (MuSiQue questions "
            "are designed to need full chains). Re-run benchmark with fixed retrieval."
        )
    print(f"\n  BM25 recall: {retrieval_recall_pct:.1f}%")
    print(f"  -> {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "retrieval_quality": {k: v for k, v in rq.items() if k != "details"},
        "retrieval_failures_sample": failures,
        "live_responses": {
            "n": len(live),
            "refused": refused_total,
            "refused_unrecoverable": refused_unrecoverable,
            "refused_recoverable": refused_recoverable,
            "records": live,
        },
        "verdict": verdict,
    }
    out_path = SCRIPT_DIR / "results" / "musique_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
