# A script for vLLM batch inference using the core LLM class.

import argparse
import json
import os
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM batch inference using the core LLM class.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSONL file.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size for vLLM.")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length.")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k for sampling.")
    parser.add_argument("--debug_limit", type=int, default=0, help="Limit the number of prompts to process for debugging. 0 means no limit.")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load prompts from the input file.
    print(f"Reading prompts from {args.input_file}...")
    prompts = []
    request_details = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            request_details.append({"request_id": data["request_id"]})
    print(f"Loaded {len(prompts)} prompts.")

    # 💡 Slice the lists if a debug limit is set.
    if args.debug_limit > 0:
        print(f"--- DEBUG MODE: Limiting to first {args.debug_limit} prompts. ---")
        prompts = prompts[:args.debug_limit]
        request_details = request_details[:args.debug_limit]

    print(f"Loaded {len(prompts)} prompts to process.")
    
    conversations = [[{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, {"role": "user", "content": p}] for p in prompts]

    # 2. Initialize the vLLM engine.
    print(f"Initializing vLLM for model: {args.model_id}...")
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
    )

    # 5. Run batch generation.
    print("Starting batch inference...")
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print("Batch inference complete.")

    # 6. Write the results to a single output JSONL file.
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, "outputs.jsonl")
    print(f"Writing output to {output_file_path}...")
    
    with open(output_file_path, "w") as f:
        for i, output in enumerate(outputs):
            result = {
                "request_id": request_details[i]["request_id"],
                "prompt": prompts[i],
                "generated_text": output.outputs[0].text,
            }
            f.write(json.dumps(result) + "\n")

    print("All tasks finished successfully.")

if __name__ == "__main__":
    main()