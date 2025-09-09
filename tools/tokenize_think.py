from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_cot_example_qwen(
    example: Dict,
    tokenizer,
):
    thinking_trajectory = example["reasoning_trace"]
    question = example["question"]
    answer = example["model_attempt"] 
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<|im_start|>think\n" + thinking_trajectory.strip() + "\n<|im_start|>answer\n" + answer.strip()
        }
    ], tokenize=False)

    # Tokenize the output text to get token IDs
    tokenized_output_all = tokenizer(text)
    tokenized_output_think = tokenizer(thinking_trajectory.strip())
    tokenized_output_answer = tokenizer(answer.strip())
    total_length = len(tokenized_output_all["input_ids"])
    think_length = len(tokenized_output_think["input_ids"])
    answer_length = len(tokenized_output_answer["input_ids"])

    
    # Return the original text and the token length
    return dict(text=text, total_length=total_length, think_length=think_length, answer_length=answer_length)

def tokenize(upload_data_path: str, num_proc: int,
                download_data_path):

    dataset = load_dataset(download_data_path, download_mode='force_redownload')
    if 'train' in dataset:
        dataset = dataset['train']
    tokenizer_name = "Qwen/Qwen2.5-32B-Instruct"
    # tokenizer_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    process_example_map = partial(process_cot_example_qwen, tokenizer=tokenizer)

    dataset = dataset.filter(lambda x: x["valid"] == 1)
    dataset = dataset.map(process_example_map, num_proc=num_proc, desc="Tokenizing SFT data")
    dataset.push_to_hub(upload_data_path, private=True)

if __name__ == "__main__":
    tokenize(download_data_path="jjzha/fs1",
                upload_data_path="jjzha/fs1-2708", 
                num_proc=50)