#!/usr/bin/env python3

import argparse
import os
import shutil
from accelerate.utils import merge_fsdp_weights, enable_fsdp_ram_efficient_loading
from transformers import AutoConfig, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Merge FSDP weights and save model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to original model directory (with config.json & safetensors)."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the FSDP sharded checkpoint directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where merged weights (checkpoint) will be stored."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help="Shard size for saving the model (e.g., 10GB, 5GB, etc.)."
    )
    args = parser.parse_args()

    # 1) copy or load+save config.json
    config = AutoConfig.from_pretrained(args.model)
    config.save_pretrained(args.output_path)

    # 2) merge
    enable_fsdp_ram_efficient_loading()
    
    merge_fsdp_weights(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path
    )
    # 3) load and re‑save in safe‑serialized, sharded format
    model = AutoModelForCausalLM.from_pretrained(
        args.output_path,
        device_map="auto",
    )
    model.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size
    )

if __name__ == "__main__":
    main()
