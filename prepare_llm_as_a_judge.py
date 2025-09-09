#!/usr/bin/env python3

import argparse
import json
import re
import os
import glob

# Constant to define the maximum number of lines per output file.
MAX_LINES_PER_FILE = 1000000000

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Transform model predictions into API calls for evaluation.")
    parser.add_argument(
        "--input_paths", type=str, nargs='+', required=True,
        help="One or more paths to JSONL prediction files or directories containing them."
    )
    parser.add_argument(
        "--benchmark_json", type=str, required=True,
        help="Path to the ground truth benchmark JSON file."
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path and base name for the output JSONL files (e.g., 'output/api_calls.jsonl')."
    )
    return parser.parse_args()

def load_benchmark(benchmark_path):
    """Loads the benchmark file and creates a dictionary mapping IDs to all possible answers."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    id_to_gold = {}
    for entry in benchmark_data:
        all_answers = entry.get('main_answers', []) + entry.get('answer_aliases', [])
        id_to_gold[entry['ID']] = all_answers
    print(f"✅ Loaded {len(id_to_gold)} entries from benchmark file.")
    return id_to_gold

def extract_response_text(item):
    """Safely extracts the main response string from various possible JSON structures."""
    # Prioritize the new 'generated_text' key.
    if 'generated_text' in item:
        return item['generated_text']
        
    try:
        return item['body']['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        pass
    
    if isinstance(item.get('message'), dict):
        return item['message'].get('content', '')
        
    return (
        item.get('completion') or
        item.get('answer') or
        item.get('response') or
        ''
    )

def extract_prediction(full_response_text):
    """
    Extracts the final answer from the full response text, preferring the \\boxed{} format.
    """
    # Search for \boxed{...} anywhere in the full text
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', full_response_text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    if '<|im_start|>answer' in full_response_text:
        return full_response_text.split('<|im_start|>answer')[1]
    
    # If not found, fall back to the last 50 tokens
    tokens = full_response_text.split()
    return ' '.join(tokens[-10:]).strip()


def process_and_yield_items(file_path, id_to_gold):
    """
    Processes a single prediction file and yields formatted items.
    It no longer writes directly to a file.
    """
    # Get the parent directory's name, which represents the model name.
    # For example, '/path/to/model-name/output.jsonl' becomes 'model-name'.
    model_name = os.path.basename(os.path.dirname(file_path))
    
    file_name_for_log = os.path.basename(file_path)
    print(f"Processing {file_name_for_log} (model ID: '{model_name}')...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠️ Warning: Skipping malformed JSON on line {line_num} in {file_name_for_log}")
                continue

            # Prioritize 'request_id' for finding the base ID.
            base_id = item.get('request_id') or item.get('custom_id') or item.get('ID') or item.get('id')
            if not base_id:
                print(f"  ⚠️ Warning: Skipping item on line {line_num} due to missing ID.")
                continue

            lookup_id = base_id.split('$')[0]
            if 'rep' in lookup_id:
                lookup_id, rep = lookup_id.split('-rep-')
            
            gold_list = id_to_gold.get(lookup_id)
            if gold_list is None:
                print(f"  ⚠️ Warning: No gold answer found for ID '{lookup_id}' (from base ID '{base_id}'). Skipping.")
                continue

            response_text = extract_response_text(item)
            if not response_text:
                print(f"  ⚠️ Warning: Could not extract response text for ID '{base_id}'. Skipping.")
                continue

            predicted_answer = extract_prediction(response_text)
            
            user_content = (
                f"gold answer: {json.dumps(gold_list)}\n"
                f"predicted answer: {json.dumps(predicted_answer)}\n\n"
                "Is the gold answer entity or value contained in the predicted answer? Respond only with 0 (no) or 1 (yes)."
            )

            # The custom_id now includes the model name from the parent folder.
            final_custom_id = f"{base_id}${model_name}"
            if 'rep' in lookup_id:
                final_custom_id = f"{base_id}-{rep}${model_name}"

            new_item = {"request_id": final_custom_id, "prompt": user_content}
            yield new_item
            
def gather_files(input_paths):
    """Gathers all .jsonl files from the provided paths."""
    all_files = []
    for path in input_paths:
        if os.path.isdir(path):
            all_files.extend(glob.glob(os.path.join(path, '*.jsonl')))
        elif os.path.isfile(path) and path.endswith('.jsonl'):
            all_files.append(path)
    return all_files

def main():
    """Main function to run the script."""
    args = parse_args()
    id_to_gold = load_benchmark(args.benchmark_json)
    
    files_to_process = gather_files(args.input_paths)
    if not files_to_process:
        print("❌ Error: No .jsonl files found in the specified input paths.")
        return

    line_counter = 0
    file_chunk_index = 1
    outfile = None
    
    base_name, ext = os.path.splitext(args.output_json)

    def open_new_chunk_file():
        nonlocal file_chunk_index
        chunk_path = f"{base_name}_part_{file_chunk_index}{ext}.jsonl"
        print(f"\nWriting to new chunk file: {chunk_path}")
        file_chunk_index += 1
        return open(chunk_path, 'w', encoding='utf-8')

    try:
        for file_path in files_to_process:
            for item_to_write in process_and_yield_items(file_path, id_to_gold):
                if outfile is None or line_counter >= MAX_LINES_PER_FILE:
                    if outfile:
                        outfile.close()
                    outfile = open_new_chunk_file()
                    line_counter = 0

                outfile.write(json.dumps(item_to_write) + "\n")
                line_counter += 1
    finally:
        if outfile:
            outfile.close()

    print(f"\n🚀 Success! Transformed data written to chunked files starting with '{base_name}_part_...'.")

if __name__ == "__main__":
    main()