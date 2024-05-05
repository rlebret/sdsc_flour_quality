"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import click
import streamlit as st
import yaml
import os

from flour.models import *  # noqa
from flour.data import FlourDataset

st.set_option("deprecation.showPyplotGlobalUse", False)


@click.command()
@click.option("--config-file", default="config.yaml")
def app(config_file):
    print(config_file)
    st.title("Flour Quality Prediction :100:")
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # define the options
    do_regression = config["regression"] if "regression" in config else False

    # Instantiate the model
    model_name = config["model_name"]
    model_params = config["hyperparameters"]  # noqa
    config["hyperparameters"]
    model = eval(f"{model_name}(**model_params)")
    # load model
    model.load(os.path.join(config["output_path"], "model.pkl"))
    print("Model loaded")

    # Load the dataset
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        # load dataset
        data = FlourDataset(
            uploaded_file,
            scaler=config["scaling_method"],
            impute_missing_values=True,
            impute_negative_values=True,
            remove_empty_columns=True,
        )
        if os.path.exists(os.path.join(config["output_path"], "scaler.pkl")):
            data.load_scaler(os.path.join(config["output_path"], "scaler.pkl"))
            print("Scaler loaded")

        # One-hot encode the categorical variables
        X = data.prepare_data_inference(
            one_hot_columns=["Production Recipe"],
            remove_columns=["Package ID", "Production Mill", "Color"],
        )
        # Make predictions
        predictions = model.predict(X)

        if do_regression:
            predictions = data.convert_to_categorical(predictions)
        data.data.insert(0, "Quality", predictions)

        tab1, tab2 = st.tabs(["Analysis", "Data"])

        with tab1:
            st.title("Analysis")
            profit = data.compute_profit(predictions)
            st.success(f"Expected profit: CHF {profit}", icon="✅")

            data.data["Quality"].value_counts().plot(
                kind="pie",
                ylabel="",
                title="Quality",
                legend=False,
                autopct="%1.1f%%",
                startangle=0,
            )
            st.pyplot()
        with tab2:
            st.title("Predictions")
            st.text("Edit the predicted quality for better results")
            st.data_editor(data.df())
            st.button("Save")


if __name__ == "__main__":
    app(standalone_mode=False)
