"""
AnchorGRPO Training — Live Reward Computation
================================================
Trains on CRAG dataset (like TruthRL) with CrestRL V2 ternary reward.
Evaluates on CRAG, NQ, HotpotQA, MuSiQue.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

CACHE_DIR = Path(SCRIPT_DIR).parent / "workdir" / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    BASE_MODEL, MODEL_CACHE_DIR, DATA_DIR, CHECKPOINT_DIR, MERGED_DIR,
    CRAG_DIR, CRAG_MAX_SAMPLES,
    GRPO_CONFIG, EPS_FLOOR, DELTA_ABSTAIN, LAMBDA_CALIB,
    ALPHA_ANCHOR,
)
from reward import (
    get_verdict, extract_confidence,
    compute_live_reward, compute_group_advantages,
)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(path=None):
    p = path or str(MODEL_CACHE_DIR / "Mistral-7B-Instruct-v0.3")
    if not Path(p).exists():
        p = BASE_MODEL
    print(f"Loading: {p}")
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


def generate(mdl, tok, prompt, temperature=0.7, max_tokens=300):
    msgs = [
        {"role": "system", "content": "You are a helpful, accurate assistant. If unsure, say so. Do not fabricate."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inp, max_new_tokens=max_tokens, temperature=temperature,
            top_p=0.9, do_sample=True, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ─── CRAG Training Data Loading (same as TruthRL) ────────────────────────────

GROUNDED_PROMPT = """Based ONLY on the following sources, answer the question.
If the sources don't contain enough information, say "I don't know".
If the sources contradict your prior knowledge, trust the sources.
Do NOT use any information not present in the sources.

Sources:
{context}

Question: {query}
Answer:"""


def load_crag_training(max_samples=None):
    """Load CRAG validation split for training (same as TruthRL)."""
    cache = CRAG_DIR / "crag_validation"
    try:
        if cache.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(cache))
        else:
            from datasets import load_dataset
            print("  Downloading CRAG...")
            ds = load_dataset("Quivr/CRAG", "crag_task_1_and_2", split="train")
            ds = ds.filter(lambda x: x.get("split", 0) == 0)
            CRAG_DIR.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache))

        samples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            query = item.get("query", "")
            answer = item.get("answer", "")
            if not query or not answer:
                continue

            search_results = item.get("search_results", [])
            context_pages = []
            if isinstance(search_results, list):
                for sr in search_results[:10]:
                    page_text = sr.get("page_snippet", "") or sr.get("page_result", "")
                    if page_text:
                        context_pages.append(page_text[:500])

            context_str = "\n\n".join([f"[Source {j+1}]: {p}" for j, p in enumerate(context_pages)])

            prompt = GROUNDED_PROMPT.format(context=context_str, query=query) if context_str else f"Answer: {query}"

            samples.append({
                "prompt": prompt,
                "expected": answer,
                "question": query,
                "domain": item.get("domain", "unknown"),
            })

        print(f"  CRAG training: {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"  CRAG load failed: {e}")
        return []


# ─── Step 1: Data Generation ─────────────────────────────────────────────────

def step_data():
    """Generate CRAG training data for GRPO (like TruthRL)."""
    print("=" * 60)
    print("DATA GENERATION — CRAG Training Set")
    print("=" * 60)

    samples = load_crag_training(max_samples=CRAG_MAX_SAMPLES)
    if not samples:
        print("ERROR: No CRAG samples loaded!")
        return

    out = DATA_DIR / "training_data.jsonl"
    with open(out, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(f"  Saved {len(samples)} training prompts to {out}")
    print("  Next step: python train.py --step train")


# ─── Step 2: Training with Live Rewards ───────────────────────────────────────

def step_train():
    """
    AnchorGRPO training with live reward computation.

    During each GRPO step:
    1. Model generates G completions per prompt
    2. For each completion, compute p_know from the model's own logits
    3. Compute CrestRL V2 reward using live p_know
    4. Compute group advantages with variance floor
    5. Update model via policy gradient
    """
    print("=" * 60)
    print("AnchorGRPO TRAINING — Live Rewards")
    print("=" * 60)

    data_path = DATA_DIR / "training_data.jsonl"
    if not data_path.exists():
        print("ERROR: No training data. Run: python train.py --step data")
        return

    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} prompts")

    # Load model
    mdl, tok = load_model()
    device = mdl.device

    # Apply LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    mdl = prepare_model_for_kbit_training(mdl)
    lora = LoraConfig(
        r=GRPO_CONFIG["lora_r"], lora_alpha=GRPO_CONFIG["lora_alpha"],
        lora_dropout=GRPO_CONFIG["lora_dropout"],
        target_modules=GRPO_CONFIG["target_modules"].split(","),
        bias="none", task_type="CAUSAL_LM",
    )
    mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()

    # ─── Custom GRPO Training Loop with Live Rewards ──────────────────────────
    #
    # TRL's GRPOTrainer calls reward_func(completions, **kwargs)
    # We need to compute rewards using the model's logits, not pre-computed values.
    #
    # Solution: the reward function captures the model via closure and computes
    # p_know from the model's logits during the forward pass.

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    # We need the prompts + expected answers for reward computation
    # Store them so the reward function can look them up
    _data = data
    _model_ref = mdl
    _tok_ref = tok

    def live_reward_func(completions, prompts=None, **kwargs):
        """
        Compute CrestRL V2 rewards LIVE during training.

        This function is called by GRPOTrainer after generating completions.
        It uses the model's own logits to compute p_know for each completion.
        """
        rewards = []

        for i, completion in enumerate(completions):
            # Get the corresponding prompt and expected answer
            idx = i % len(_data)
            prompt = _data[idx]["prompt"]
            expected = _data[idx]["expected"]

            # Compute reward using the model's logits for p_know
            try:
                reward = compute_live_reward(
                    model=_model_ref,
                    tokenizer=_tok_ref,
                    query=prompt,
                    completion=completion,
                    expected=expected,
                    device=device,
                    lambda_calib=LAMBDA_CALIB,
                    lambda_anchor=ALPHA_ANCHOR,
                    eps_floor=EPS_FLOOR,
                )
            except Exception as e:
                # Fallback: simple binary reward if logit computation fails
                from reward import get_verdict
                v = get_verdict(completion, expected)
                reward = 1.0 if v == "correct" else -1.0 if v == "hallucination" else 0.0

            rewards.append(reward)

        return rewards

    # Create dataset with prompts
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])

    # GRPO Config
    args = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=GRPO_CONFIG["num_epochs"],
        per_device_train_batch_size=GRPO_CONFIG["batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=GRPO_CONFIG["learning_rate"],
        lr_scheduler_type=GRPO_CONFIG["lr_scheduler"],
        warmup_ratio=GRPO_CONFIG["warmup_ratio"],
        max_completion_length=GRPO_CONFIG["max_seq_length"],
        num_generations=GRPO_CONFIG["num_generations"],
        beta=GRPO_CONFIG["beta"],
        max_grad_norm=GRPO_CONFIG["max_grad_norm"],
        max_steps=200,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="paged_adamw_8bit",
    )

    print("\nStarting AnchorGRPO training (live rewards)...")
    t0 = time.time()
    trainer = GRPOTrainer(
        model=mdl, args=args, train_dataset=dataset,
        reward_funcs=live_reward_func, processing_class=tok,
    )
    trainer.train()
    elapsed = time.time() - t0

    out = str(CHECKPOINT_DIR)
    mdl.save_pretrained(out)
    tok.save_pretrained(out)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {elapsed/60:.1f} min")
    print(f"Checkpoint: {out}")
    print(f"Next: python train.py --step merge")


def step_merge():
    """Merge LoRA weights into full model."""
    from peft import PeftModel

    base = str(MODEL_CACHE_DIR / "Mistral-7B-Instruct-v0.3")
    if not Path(base).exists():
        base = BASE_MODEL
    out = str(MERGED_DIR)

    print(f"Merging LoRA into full model...")
    print(f"  Base: {base}")
    print(f"  LoRA: {CHECKPOINT_DIR}")
    print(f"  Output: {out}")

    mdl = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    mdl = PeftModel.from_pretrained(mdl, str(CHECKPOINT_DIR))
    mdl = mdl.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    mdl.save_pretrained(out)

    # Save tokenizer (from checkpoint if available, else from base)
    try:
        tok = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)
    except Exception:
        print("  Checkpoint tokenizer not found, using base tokenizer")
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.save_pretrained(out)
    print(f"  Merged model saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["data", "train", "merge"], required=True)
    args = p.parse_args()

    {"data": step_data, "train": step_train, "merge": step_merge}[args.step]()
