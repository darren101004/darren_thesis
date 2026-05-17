import argparse
from pathlib import Path

from datasets import Dataset
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val dataset from NSFW prompt CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../data/nsfw_prompts.csv",
        help="Path to CSV with columns: id, original_text, rewritten_text, category",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/dataset",
        help="Output directory for HuggingFace dataset (train/val split).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = ["id", "original_text", "rewritten_text", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].dropna(subset=["original_text", "rewritten_text"]).copy()
    df["original_text"] = df["original_text"].astype(str).str.strip()
    df["rewritten_text"] = df["rewritten_text"].astype(str).str.strip()
    df = df[(df["original_text"] != "") & (df["rewritten_text"] != "")]

    # Remove exact duplicated pairs to reduce overfitting.
    df = df.drop_duplicates(subset=["original_text", "rewritten_text"]).reset_index(drop=True)

    records = df.to_dict(orient="records")
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)

    split.save_to_disk(str(output_dir))
    print(f"Saved dataset to: {output_dir.resolve()}")
    print(f"Train size: {len(split['train'])}")
    print(f"Val size:   {len(split['test'])}")


if __name__ == "__main__":
    main()
