"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import argparse
import yaml
import os
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
)
import numpy as np

from flour.models import *  # noqa
from flour.data import FlourDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    # load the config file
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    # Check if oversampling and regression are enabled
    do_oversample = config["oversample"] if "oversample" in config else False
    do_regression = config["regression"] if "regression" in config else False
    do_cross_validation = (
        config["cross_validation"] if "cross_validation" in config else False
    )

    # load the dataset
    data = FlourDataset(config["data_path"], scaler=config["scaling_method"])

    # Instantiate the model
    model_name = config["model_name"]
    model_params = config["hyperparameters"]
    model = eval(f"{model_name}(**model_params)")
    print(model.hyperparameters())

    if do_cross_validation:
        # Perform cross-validation
        y_tests, y_preds = model.cross_validate(
            data,
            regression=do_regression,
            oversample=do_oversample,
            random_state=config["random_state"],
        )
        f1_scores = []
        for k, (y_test, y_pred) in enumerate(zip(y_tests, y_preds)):
            if do_regression:
                mse = mean_squared_error(y_test, y_pred)
                print(f"{k}-fold -- Mean Squared Error:", mse)
                # Convert the numerical target variable to categorical
                y_pred = data.convert_to_categorical(y_pred)
                y_test = data.convert_to_categorical(y_test)
            f1 = f1_score(y_test, y_pred, average="macro")
            f1_scores.append(f1)
        f1 = np.mean(f1_scores)
        print("Cross-validation F1-score:", f1)
    else:
        # Split the dataset into features and target
        X_train, X_test, y_train, y_test = data.split_train_test(
            test_size=config["test_size"],
            random_state=config["random_state"],
            oversample=do_oversample,
            regression=do_regression,
        )
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)

        # Convert the numerical target variable to categorical
        if do_regression:
            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error:", mse)
            # Convert the numerical target variable to categorical
            y_pred = data.convert_to_categorical(y_pred)
            y_test = data.convert_to_categorical(y_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        print("F1 Score:", f1)

    # create output directory if it does not exist
    os.makedirs(config["output_path"], exist_ok=True)
    # save f1 score to file
    with open(os.path.join(config["output_path"], "f1.txt"), "w") as f:
        f.write(str(f1))

    # Prepare the data for full training
    X, y = data.prepare_data(
        train=True,
        regression=do_regression,
        oversample=do_oversample,
        random_state=config["random_state"],
    )

    # Train the model on the full dataset
    model.reset()
    model.fit(X, y)
    # save scaler
    data.save_scaler(os.path.join(config["output_path"], "scaler.pkl"))
    # save the model
    model.save(os.path.join(config["output_path"], "model.pkl"))
    print("Model saved")
