#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import HfApi, login

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "repo_name",
        help="Name of the model repository to create/use on HF (e.g. 'Qwen2.5-3B-Instruct-rt')."
    )
    parser.add_argument(
        "folder_path",
        help="Path to the local checkpoint folder to upload."
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (defaults to $HF_TOKEN env var)."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.token:
        raise RuntimeError("Hugging Face token not provided. Use --token or set HF_TOKEN.")

    # Log in
    login(token=args.token)
    api = HfApi()

    # Upload
    api.upload_large_folder(
        folder_path=args.folder_path,
        repo_id=f"jjzha/{args.repo_name}",
        repo_type="model",
        private=args.private,
    )

    print(f"[INFO] Finished uploading '{args.repo_name}' from '{args.folder_path}'.")

if __name__ == "__main__":
    main()
