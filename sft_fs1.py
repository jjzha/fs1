# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import load_dataset
from transformers import AutoTokenizer
import huggingface_hub
from functools import partial
import math
from pathlib import Path
from pprint import pprint
import os
from env_utils import print_slurm_env
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    DataCollatorForCompletionOnlyLM
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # LUMI dedicated stuff
    ################    

    # Setup distributed environment
    torch.multiprocessing.set_start_method(
        # Workaround "fork" not being safe with Slingshot 11 when using multiple
        # PyTorch DataLoader workers
        "spawn"
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    os.sched_setaffinity(
        # Set CPU bindings based on LOCAL_RANK which is also used to set GPU device by accelerate
        0,
        LUMI_GPU_CPU_map[local_rank],  # Set CPU binding for the current process (0)
    )
    print_slurm_env()  # Print SLURM environment

    ################
    # Model init kwargs & Tokenizer
    ################
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)

    if "Qwen" in model_config.model_name_or_path:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"
    elif "Smol" in model_config.model_name_or_path:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<empty_output>"
    elif "Llama" in model_config.model_name_or_path:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    else:
        instruction_template = "<start_of_turn>user"
        response_template = "<start_of_turn>model\n"
        tokenizer.pad_token = "<pad>"

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    dataset = load_dataset(script_args.dataset_name)
    max_seq_length = training_args.max_seq_length
    # make sure to only take items smaller than 8,192 tokens and only correct traces (label==1)
    filtered_dataset = dataset.filter(lambda example: example['total_length'] <= max_seq_length).filter(lambda example: example['label'] == 1)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=filtered_dataset['train'],
        eval_dataset=filtered_dataset['test'] if 'test' in filtered_dataset else filtered_dataset['train'],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    cfg = trainer.model.config
    cfg.save_pretrained(training_args.output_dir)

    trainer.accelerator.wait_for_everyone()
