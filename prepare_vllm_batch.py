#!/usr/bin/env python3

import argparse
import json
import os
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Huggingface dataset and extract a column to create a JSONL file of prompts for vLLM offline inference."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="jjzha/fs1-benchmark-2708",
        help="Huggingface dataset identifier"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to load (default: test)"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="question",
        help="Name of the column to extract for the prompt"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write the output JSONL file"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the specified dataset split
    ds = load_dataset(args.dataset, split=args.split)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write prompts to a single JSONL file
    count = 0
    with open(args.output_file, 'w') as f:
        for item in ds:
            # vLLM's offline inference expects a 'prompt' key by default
            content = item.get(args.column, "")
            request_id = item.get("ID", "")
            
            payload = {
                "request_id": f"{request_id}",
                "prompt": content + ' Put your final answer in \\boxed{}.'
            }
            f.write(json.dumps(payload) + "\n")
            count += 1

    print(f"✅ Transformed {count} prompts into '{args.output_file}'")

if __name__ == "__main__":
    main()