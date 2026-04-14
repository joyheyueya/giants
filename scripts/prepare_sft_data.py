"""Download GiantsBench from the Hugging Face Hub and materialize parquet splits.

Example:

    python scripts/prepare_sft_data.py \\
        --dataset giants2026/GiantsBench-train \\
        --output-dir data/insight_anticipation_sft

The script writes ``train.parquet`` and ``test.parquet`` inside ``--output-dir``.
If the source dataset does not already define a test split, a deterministic
3%% holdout is carved out with seed 42.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="giants2026/GiantsBench-train",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to download from the source dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory to write train.parquet and test.parquet into.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.03,
        help="Fraction used for the test split when the source has no test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for dataset loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} (split={args.split})...")
    dataset = datasets.load_dataset(args.dataset, num_proc=args.num_proc, split=args.split)

    if "test" not in dataset:
        dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    print(dataset)

    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    dataset["train"].to_parquet(train_path)
    dataset["test"].to_parquet(test_path)

    print(f"Wrote {train_path} ({len(dataset['train'])} rows)")
    print(f"Wrote {test_path} ({len(dataset['test'])} rows)")


if __name__ == "__main__":
    main()
