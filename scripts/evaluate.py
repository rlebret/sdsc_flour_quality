"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import argparse
import yaml
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from flour.models import *  # noqa
from flour.data import FlourDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, help="Path to the config file")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    # load the config file
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    # define the options
    do_regression = config["regression"] if "regression" in config else False

    # Load the dataset
    data = FlourDataset(
        args.test_file,
        remove_empty_columns=True,
        impute_missing_values=True,
        impute_negative_values=True,
        scaler=config["scaling_method"],
    )
    if os.path.exists(os.path.join(config["output_path"], "scaler.pkl")):
        data.load_scaler(os.path.join(config["output_path"], "scaler.pkl"))
        print("Scaler loaded")
    # Load the model

    model_name = config["model_name"]
    model_params = config["hyperparameters"]
    model = eval(f"{model_name}(**model_params)")
    model.load(os.path.join(config["output_path"], "model.pkl"))
    print(model.hyperparameters())

    # Do the predictions
    X_test, y_test = data.prepare_data(
        remove_columns=["Package ID", "Production Mill", "Color"],
        one_hot_columns=["Production Recipe"],
        regression=do_regression,
    )
    y_pred = model.predict(X_test)

    # Save the evaluation metrics
    desired_order = ["Low", "Average", "High"]
    if do_regression:
        y_test = data.convert_to_categorical(y_test)
        y_pred = data.convert_to_categorical(y_pred)
    classification_report = classification_report(y_test, y_pred)
    with open(
        os.path.join(config["output_path"], "classification_report.txt"), "w"
    ) as f:
        f.write(classification_report)

    # Save the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=desired_order)

    ## Create a heatmap visualization
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=desired_order,
        yticklabels=desired_order,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # Save the confusion matrix
    fig.savefig(os.path.join(config["output_path"], "confusion_matrix.png"))
