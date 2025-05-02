#!/usr/bin/env python3
"""
Generate answers for every row in a HuggingFace dataset, with optional
    – budget-forcing “think-then-answer” generation
    – source_of_data filtering
    – resume-from-ID checkpointing
"""

import argparse
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Budget-forcing helper
# ──────────────────────────────────────────────────────────────────────────────
def budget_forcing_answer(model, tok, stop_token_ids, prompt_text, budgets):
    history = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt_text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SPECIAL_PATTERNS = ("<|im_start|>answer",)  # marker the model emits when it wants to answer
    MARGIN           = 32                      # allow the marker when ≤32 tokens remain

    max_budget = max(budgets)                  # treat the largest budget as the hard cap
    injected_answer_token = False

    while True:
        used      = len(tok(history, return_tensors="pt")["input_ids"][0])
        remaining = max_budget - used
        if remaining <= 0:                     # bucket exhausted
            break

        chunk = model.generate(
            history,
            sampling_params=SamplingParams(
                max_tokens          = remaining,
                stop_token_ids      = stop_token_ids,
                skip_special_tokens = False,
                temperature         = 0.0,
                repetition_penalty  = 1.05,
            ),
        )[0].outputs[0].text

        # ── cut away premature markers ───────────────────────────────────────
        early_stop = False
        for pat in SPECIAL_PATTERNS:
            pos = chunk.find(pat)
            if pos != -1:
                chunk      = chunk[:pos].rstrip()
                early_stop = True
                break

        history += chunk
        remaining_after_chunk = max_budget - len(tok(history, return_tensors="pt")["input_ids"][0])

        # keep thinking if we saw a marker but are still far from the cap
        if early_stop and remaining_after_chunk > MARGIN:
            history += " Wait,"
            continue

        # otherwise start the real answer
        history += "\n<|im_start|>answer\n"
        injected_answer_token = True
        break

    if not injected_answer_token:              # safety net
        history += "\n<|im_start|>answer\n"

    final = model.generate(
        history,
        sampling_params=SamplingParams(
            max_tokens          = 256,
            stop_token_ids      = stop_token_ids,
            skip_special_tokens = False,
            temperature         = 0.0,
        ),
    )
    history += final[0].outputs[0].text
    return history.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Simple one-shot helper (no budgets)
# ──────────────────────────────────────────────────────────────────────────────
def simple_answer(model, stop_token_ids, prompt):
    resp = model.generate(
        prompt,
        sampling_params=SamplingParams(
            max_tokens          = 8192,
            min_tokens          = 0,
            stop_token_ids      = stop_token_ids,
            skip_special_tokens = False,
            temperature         = 0.0,
            repetition_penalty  = 1.05,
        ),
    )
    return resp[0].outputs[0].text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate answers row-by-row")
    parser.add_argument("--dataset",         required=True,  help="HF dataset name (e.g. squad)")
    parser.add_argument("--split",           default="train")
    parser.add_argument("--column",          default="question",
                        help="Dataset column to use as the user prompt")
    parser.add_argument("--model",           default="simplescaling/s1")
    parser.add_argument("--tp-size",         type=int, default=2)
    parser.add_argument("--budgets",         type=int, nargs="*", default=None,
                        help="Token budgets for budget-forcing (e.g. 512 2048 8000)")
    parser.add_argument("--output",          default="output.jsonl")
    parser.add_argument("-n", "--max-examples", type=int)
    parser.add_argument("--max-num-seqs",    type=int)
    parser.add_argument("--source-filter",   nargs="+",
                        help="Process only rows whose source_of_data is in this list")
    # NEW: resume-from-ID flag
    parser.add_argument("--resume-from-id",  metavar="ID",
                        help="Skip rows with ID ≤ this value then resume")

    args = parser.parse_args()

    # load model & tokenizer
    model = LLM(args.model,
                tensor_parallel_size=args.tp_size,
                max_num_seqs=args.max_num_seqs)
    tok   = AutoTokenizer.from_pretrained(args.model)
    stop_token_ids = tok("<|im_end|>")["input_ids"]

    # load dataset split
    ds = load_dataset(args.dataset, split=args.split)

    with open(args.output, "a", encoding="utf-8") as fout:
        for idx, row in enumerate(ds):
            if args.max_examples is not None and idx >= args.max_examples:
                break

            # optional source_of_data filter
            if args.source_filter and row.get("source_of_data") not in args.source_filter:
                continue

            # optional resume-from-ID filter
            if args.resume_from_id is not None and int(str(row["ID"]).split("_")[-1]) < int(args.resume_from_id.split("_")[-1]):
                continue

            # build prompt
            q = row[args.column]
            prompt = (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{q}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            # generate answer
            if args.budgets:
                answer = budget_forcing_answer(model, tok, stop_token_ids, q, args.budgets)
            else:
                answer = simple_answer(model, stop_token_ids, prompt)

            # write result
            fout.write(
                json.dumps(
                    {
                        "id":       row["ID"],
                        "question": q,
                        "answer":   answer,
                        "source":   row.get("source_of_data"),
                    },
                    ensure_ascii=False
                ) + "\n"
            )

    total = args.max_examples if args.max_examples is not None else len(ds)
    print(f"✅ Done! Processed {min(total, len(ds))} samples → {args.output}")


if __name__ == "__main__":
    main()
