import argparse
import json
import os
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Huggingface dataset and create a JSONL file of prompts for vLLM offline inference."
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
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to repeat each prompt"
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
    total_prompts = 0
    with open(args.output_file, 'w') as f:
        # Iterate over each original item in the dataset
        for item in ds:
            content = item.get(args.column, "")
            original_id = item.get("ID", "")
            prompt_text = content + ' Put your final answer in \\boxed{}.'

            for i in range(args.repetitions):
                payload = {
                    # Create a unique ID for each repetition, e.g., "ID-123-rep-0"
                    "request_id": f"{original_id}-rep-{i}",
                    "prompt": prompt_text #+ " Think step-by-step."
                }
                f.write(json.dumps(payload) + "\n")
                total_prompts += 1

    print(f"✅ Transformed {len(ds)} prompts into {total_prompts} total requests in '{args.output_file}'")

if __name__ == "__main__":
    main()