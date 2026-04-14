"""Build GRPO-ready parquet splits from the GiantsBench Hugging Face dataset.

The source dataset (e.g. ``giants2026/GiantsBench-train``) contains columns
``query``, ``completion``, and ``pair_id``. ``completion`` is a long string that
wraps the target insight inside ``<insight>...</insight>`` tags. verl's RL
trainer expects each row to look like::

    {
        "data_source": "",
        "prompt": [{"role": "user", "content": <query>}],
        "ability": "insight",
        "reward_model": {"style": "rule", "ground_truth": <insight text>},
        "extra_info": {"pair_id": ..., "split": ..., "index": ...}
    }

This script performs that transformation and writes ``train.parquet`` /
``test.parquet`` inside ``--output-dir``. If the source has no ``test`` split
a deterministic holdout is carved with seed 42.

Example:

    python scripts/prepare_rl_data.py \\
        --dataset giants2026/GiantsBench-train \\
        --output-dir data/insight_anticipation_grpo
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict

import datasets

INSIGHT_PATTERN = re.compile(r"<insight>(.*?)</insight>", re.DOTALL)


def extract_insight(text: str) -> str:
    matches = INSIGHT_PATTERN.findall(text or "")
    return matches[0].strip() if matches else ""


def make_map_fn(split: str):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example.pop("query")
        raw_completion = example.pop("completion")
        answer = extract_insight(raw_completion)
        return {
            "data_source": "",
            "prompt": [{"role": "user", "content": question}],
            "ability": "insight",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "pair_id": example.get("pair_id"),
                "split": split,
                "index": idx,
            },
        }

    return process_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="giants2026/GiantsBench-train",
                        help="Hugging Face dataset repo id.")
    parser.add_argument("--split", default="train",
                        help="Split to download from the source dataset.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write train.parquet and test.parquet into.")
    parser.add_argument("--test-size", type=float, default=0.03,
                        help="Holdout fraction when the source has no test split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train/test split.")
    parser.add_argument("--num-proc", type=int, default=os.cpu_count() or 1,
                        help="Worker processes for loading and mapping.")
    parser.add_argument("--drop-empty-insights", action="store_true",
                        help="Filter out rows where no <insight>...</insight> block was found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} (split={args.split})...")
    dataset = datasets.load_dataset(args.dataset, num_proc=args.num_proc, split=args.split)

    if "test" not in dataset:
        dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    print(dataset)

    processed = {}
    for split_name, split_data in dataset.items():
        processed[split_name] = split_data.map(
            function=make_map_fn(split_name),
            with_indices=True,
            num_proc=args.num_proc,
        )

        if args.drop_empty_insights:
            before = len(processed[split_name])
            processed[split_name] = processed[split_name].filter(
                lambda row: bool(row["reward_model"]["ground_truth"]),
                num_proc=args.num_proc,
            )
            after = len(processed[split_name])
            if before != after:
                print(f"[{split_name}] dropped {before - after} rows with no <insight> block.")

    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    processed["train"].to_parquet(train_path)
    processed["test"].to_parquet(test_path)

    print(f"Wrote {train_path} ({len(processed['train'])} rows)")
    print(f"Wrote {test_path} ({len(processed['test'])} rows)")


if __name__ == "__main__":
    main()
