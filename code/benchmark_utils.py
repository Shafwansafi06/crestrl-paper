"""
Benchmark Utilities — Shared across NQ, HotpotQA, MuSiQue runners
===================================================================
"""

import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_CACHE_DIR, BASE_MODEL


GROUNDED_PROMPT = """Based ONLY on the following sources, answer the question.
If the sources don't contain enough information, say "I don't know".
If the sources contradict your prior knowledge, trust the sources.
Do NOT use any information not present in the sources.

Sources:
{context}

Question: {query}
Answer:"""

NO_CONTEXT_PROMPT = """Answer the following question from your knowledge.
If you are unsure, say "I don't know".

Question: {query}
Answer:"""

ABSTAIN_PHRASES = [
    "i don't know", "i'm not sure", "i cannot", "i can't",
    "i'm unable", "not sure", "uncertain", "unclear",
    "cannot verify", "cannot confirm", "insufficient",
    "no information", "not enough information", "don't have",
    "cannot determine", "cannot answer", "no reliable",
]


def is_refusal(response):
    rl = response.lower()
    return any(p in rl for p in ABSTAIN_PHRASES)


def evaluate_response(response, truth):
    rl = response.lower().strip()
    tl = truth.lower().strip()
    refused = is_refusal(response)

    if tl in rl:
        correct = True
    else:
        truth_toks = [t for t in tl.split() if len(t) > 3]
        if len(truth_toks) == 0:
            truth_toks = tl.split()
        matched = sum(1 for t in truth_toks if t in rl)
        correct = matched >= max(1, len(truth_toks) // 2)

    hallucinated = not correct and not refused
    truthfulness = (1 if correct else 0) + (1 if refused else 0) - (1 if hallucinated else 0)

    return {
        "correct": correct,
        "refused": refused,
        "hallucinated": hallucinated,
        "truthfulness": truthfulness,
    }


def retrieve_passages_bm25(query, passages, top_k=5):
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [passages[i] for i in top_idx if scores[i] > 0][:top_k]
    except ImportError:
        return passages[:top_k]


def load_model(path=None, label="Model"):
    p = path or str(MODEL_CACHE_DIR / "Mistral-7B-Instruct-v0.3")
    if not Path(p).exists():
        p = BASE_MODEL
    print(f"Loading {label}: {p}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return mdl, tok


def run_benchmark(mdl, tok, samples, label="Model", dataset_name="Dataset"):
    print(f"\n{'='*60}")
    print(f"{dataset_name} BENCHMARK — {label} ({len(samples)} samples)")
    print(f"{'='*60}")

    results, domain_stats = [], {}
    t_correct = t_refused = t_halluc = t_truth = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        context = s.get("context", [])
        if isinstance(context, list) and len(context) > 0:
            passages = retrieve_passages_bm25(s["question"], context, top_k=5)
            context_str = "\n\n".join([f"[Source {j+1}]: {p}" for j, p in enumerate(passages)])
        elif isinstance(context, str) and len(context) > 50:
            context_str = context[:2000]
        else:
            context_str = ""

        if context_str and len(context_str) > 50:
            prompt = GROUNDED_PROMPT.format(context=context_str, query=s["question"])
        else:
            prompt = NO_CONTEXT_PROMPT.format(query=s["question"])

        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(
                    **inp, max_new_tokens=300, temperature=0.3, top_p=0.9,
                    do_sample=True, pad_token_id=tok.eos_token_id,
                )
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            resp = f"ERROR: {e}"
        torch.cuda.empty_cache()

        ev = evaluate_response(resp, s["answer"])
        results.append({**s, "response": resp, **ev, "model": label})

        if ev["correct"]: t_correct += 1
        if ev["refused"]: t_refused += 1
        if ev["hallucinated"]: t_halluc += 1
        t_truth += ev["truthfulness"]

        d = s.get("domain", "unknown")
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "correct": 0, "refused": 0, "hallucinated": 0, "truthfulness": 0}
        domain_stats[d]["total"] += 1
        domain_stats[d]["correct"] += int(ev["correct"])
        domain_stats[d]["refused"] += int(ev["refused"])
        domain_stats[d]["hallucinated"] += int(ev["hallucinated"])
        domain_stats[d]["truthfulness"] += ev["truthfulness"]

        if (i + 1) % 100 == 0 or i == 0:
            n = i + 1
            elapsed = time.time() - t0
            rate = n / elapsed
            eta = (len(samples) - n) / rate / 60 if rate > 0 else 0
            print(f"  [{n}/{len(samples)}] acc={t_correct/n*100:.1f}% "
                  f"truth={t_truth/n*100:.1f}% halluc={t_halluc/n*100:.1f}% "
                  f"{elapsed:.0f}s (ETA {eta:.0f}min)")

    n = len(results)
    elapsed = time.time() - t0
    summary = {
        "model": label,
        "dataset": dataset_name,
        "total": n,
        "accuracy": t_correct / n * 100 if n else 0,
        "hallucination_rate": t_halluc / n * 100 if n else 0,
        "refusal_rate": t_refused / n * 100 if n else 0,
        "truthfulness": t_truth / n * 100 if n else 0,
        "time_seconds": elapsed,
        "domain_breakdown": {
            d: {
                "accuracy": s["correct"] / s["total"] * 100,
                "hallucination_rate": s["hallucinated"] / s["total"] * 100,
                "refusal_rate": s["refused"] / s["total"] * 100,
                "truthfulness": s["truthfulness"] / s["total"] * 100,
                "count": s["total"],
            }
            for d, s in domain_stats.items()
        },
    }

    print(f"\n  Accuracy:      {summary['accuracy']:.1f}%")
    print(f"  Hallucination: {summary['hallucination_rate']:.1f}%")
    print(f"  Refusal:       {summary['refusal_rate']:.1f}%")
    print(f"  Truthfulness:  {summary['truthfulness']:.1f}%")

    return {"summary": summary, "results": results}
