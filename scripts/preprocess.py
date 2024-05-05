"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import argparse
from flour.data import FlourDataset, FLOAT_COLUMNS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3,
        help="Z-score threshold for removing outliers",
    )
    parser.add_argument(
        "--remove-empty-rows",
        action="store_true",
        help="Remove rows with empty values",
    )
    parser.add_argument(
        "--remove-negative-rows",
        action="store_true",
        help="Remove rows with negative values",
    )
    parser.add_argument(
        "--impute-missing-values",
        action="store_true",
        help="Impute missing values with the mean",
    )
    parser.add_argument(
        "--impute-negative-values",
        action="store_true",
        help="Impute negative values with the mean",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.csv",
    )
    args = parser.parse_args()
    dataset = FlourDataset(
        args.data_path,
        z_threshold=args.z_threshold,
        remove_empty_rows=args.remove_empty_rows,
        remove_negative_rows=args.remove_negative_rows,
        impute_missing_values=args.impute_missing_values,
        impute_negative_values=args.impute_negative_values,
        remove_outlier_columns=FLOAT_COLUMNS,
        remove_columns=["Package ID", "Production Mill", "Color"],
        one_hot_columns=["Production Recipe"],
    )
    print(dataset.data.head())
    print(dataset.data.describe())
    dataset.data.to_csv(args.output_path, index=False)
