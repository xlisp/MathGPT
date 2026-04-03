"""
Download HuggingFace datasets locally for offline use on servers without internet.

Usage:
    # Step 1: Run this script locally (where you have internet)
    python scripts/download_hf_datasets.py --output_dir data/hf_datasets

    # Step 2: Upload to the A800 server
    scp -r data/hf_datasets/ root@<A800_IP>:/mnt/openclaw/MathGPT/data/hf_datasets/

    # Step 3: Run training with --offline flag
    python -m scripts.chat_sft --offline ...
"""

import argparse
import os
from datasets import load_dataset

DATASETS = [
    # (repo_id, subset, splits)
    ("HuggingFaceTB/smol-smoltalk", None, ["train", "test"]),
    ("cais/mmlu", "all", ["auxiliary_train", "test"]),
    ("openai/gsm8k", "main", ["train", "test"]),
    ("allenai/ai2_arc", "ARC-Easy", ["train", "validation", "test"]),
    ("allenai/ai2_arc", "ARC-Challenge", ["train", "validation", "test"]),
    ("openai/openai_humaneval", None, ["test"]),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/hf_datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for repo_id, subset, splits in DATASETS:
        # e.g. "smol-smoltalk", "mmlu", "gsm8k", "ai2_arc/ARC-Easy"
        name = repo_id.split("/")[-1]
        # For datasets with multiple subsets (e.g. ARC), include subset in path
        if name == "ai2_arc":
            save_path = os.path.join(args.output_dir, name, subset)
        else:
            save_path = os.path.join(args.output_dir, name)
        print(f"Downloading {repo_id} ({subset or 'default'}) -> {save_path}")

        for split in splits:
            split_path = os.path.join(save_path, split)
            if os.path.exists(split_path):
                print(f"  {split} already exists, skipping")
                continue
            print(f"  downloading split={split} ...")
            if subset:
                ds = load_dataset(repo_id, subset, split=split)
            else:
                ds = load_dataset(repo_id, split=split)
            ds.save_to_disk(split_path)
            print(f"  saved {len(ds)} rows to {split_path}")

    print("\nDone! Upload data/hf_datasets/ to your server, then run training with --offline flag.")

if __name__ == "__main__":
    main()
