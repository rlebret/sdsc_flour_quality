"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import argparse
import json

from flour.models import *  # noqa
from flour.data import FlourDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the dataset")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations")
    parser.add_argument(
        "--model_name", type=str, default="RandomForestFlour", help="Model name"
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        choices=["standard", "mixmax", "none"],
        default="none",
        help="Scaling method",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--regression", action="store_true", help="Regression task")
    parser.add_argument("--output_path", type=str, default="hyperparameters.json")
    args = parser.parse_args()
    print(args)
    # Load the dataset
    data = FlourDataset(args.data_path, scaler=args.scaling_method)
    # Split the dataset into features and target
    X_train, _, y_train, _ = data.split_train_test(
        test_size=args.test_size,
        random_state=args.random_state,
        regression=args.regression,
    )

    # Instantiate the model
    model = eval(f"{args.model_name}")
    # Perform a randomized search for the hyperparameters
    best_parameters = model.hyperparameters_randomized_search(
        X_train, y_train, n_iter=args.n_iter
    )
    # Save the best hyperparameters
    with open(args.output_path, "w") as f:
        json.dump(best_parameters, f)
