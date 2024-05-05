"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path
from flour.data import FlourDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data",
    )
    args = parser.parse_args()
    data = FlourDataset(
        args.data_path,
        remove_empty_columns=True,
    )
    X = data.df()
    X_train, X_test = train_test_split(
        X,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=X["Quality"],
    )
    # save the files
    X_train.to_csv(Path(args.output_path) / "train.csv", index=False)
    X_test.to_csv(Path(args.output_path) / "test.csv", index=False)
    X_test.drop(columns=["Quality"]).to_csv(
        Path(args.output_path) / "test_no_labels.csv", index=False
    )
    print(f"Train and test files saved in {args.output_path}")
