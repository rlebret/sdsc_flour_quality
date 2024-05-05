# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
import random
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


# Define the price dictionary to map categories to numerical values
price_map = {"Low": 1.2, "Average": 2.5, "High": 5.0}

# Assuming price_map is a dictionary mapping categories to prices (from previous step)
# Invert the price_map to get a dictionary mapping prices to categories
category_map = {v: k for k, v in price_map.items()}

# Define thresholds based on price ranges (adjust as needed)
low_threshold = (
    price_map["Low"] + (price_map["Average"] - price_map["Low"]) / 2
)  # Middle point of low-average range
high_threshold = (
    price_map["Average"] + (price_map["High"] - price_map["Average"]) / 2
)  # Middle point of average-high range

FLOAT_COLUMNS = [
    "Gluten Content (%)",
    "Dough Elasticity Index",
    "Dampening Time (hours)",
    "Package Weight (g)",
    "Ash content (%)",
    "Moisture (%)",
    "Starch Content (%)",
    "Package Volume (cm3)",
    "Proteins (g)/100g",
]

CATEGORICAL_COLUMNS = [
    "Production Recipe",
]


def get_category(predicted_quality):
    if predicted_quality < low_threshold:
        return category_map[price_map["Low"]]  # "low" category
    elif predicted_quality < high_threshold:
        return category_map[price_map["Average"]]  # "average" category
    else:
        return category_map[price_map["High"]]  # "high" category


class FlourDataset:
    def __init__(
        self,
        data_path,
        z_threshold=3,
        remove_empty_rows=False,
        remove_empty_columns=True,
        remove_columns=None,
        remove_negative_rows=False,
        remove_outlier_columns=None,
        impute_missing_values=False,
        impute_negative_values=False,
        one_hot_columns=None,
        scaler=None,
    ):
        """
        Initializes the dataset.

        parameters:
        - data_path: path to the dataset
        - z_threshold: z-score threshold for removing outliers
        - remove_empty_rows: remove rows with empty values
        - remove_empty_columns: remove columns with empty values
        - remove_negative_rows: remove rows with negative values
        - remove_outliers: remove rows with z-score outliers
        - impute_missing_values: impute missing values with the mean
        - impute_negative_values: impute negative values with the mean
        - scaler: scaler to scale the features
        """
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        assert (
            impute_missing_values + remove_negative_rows <= 1
        ), "Cannot impute missing values and remove negative rows at the same time"
        assert (
            impute_negative_values + remove_empty_rows <= 1
        ), "Cannot impute negative values and remove empty rows at the same time"

        if remove_empty_columns:
            self.data.dropna(how="all", axis=1, inplace=True)
        if remove_columns is not None:
            self.data.drop(columns=remove_columns, inplace=True)
        if impute_missing_values:
            self.impute_missing_values(impute_negative_values)
        if remove_empty_rows:
            self.data.dropna(inplace=True)
        if remove_negative_rows:
            self.remove_negative_rows()
        if remove_outlier_columns is not None:
            self.remove_outliers(remove_outlier_columns, z_threshold)
        if one_hot_columns is not None:
            self.one_hot_encode(one_hot_columns)

    def df(self):
        return self.data

    def target(self, name="Quality"):
        return self.data[name]

    def impute_missing_values(self, impute_negative_values=False):
        """
        Imputes missing values with the mean.
        """
        float_columns = FLOAT_COLUMNS
        if impute_negative_values:
            self.data[float_columns] = self.data[float_columns].map(
                lambda x: np.nan if x < 0 else x
            )
        self.data[float_columns] = self.data[float_columns].apply(
            lambda x: x.fillna(x.mean()), axis=0
        )

    def remove_negative_rows(self):
        """
        Removes rows with negative values in any numerical column.
        """
        self.data = self.data[
            self.data.select_dtypes(include=[np.number]).ge(0).all(axis=1)
        ]  # >= 0 for all numerical columns

    def remove_outliers(self, columns, z_threshold=3):
        """
        Removes rows with values that are z-score outliers.
        """
        self.data = self.data[
            (np.abs(zscore(self.data[columns])) < z_threshold).all(axis=1)
        ]

    def one_hot_encode(self, columns):
        """
        One-hot encodes the specified columns.
        """
        self.data = pd.get_dummies(self.data, columns=columns)

    def split_train_test(
        self,
        test_size=0.2,
        random_state=42,
        oversample=False,
        regression=False,
        one_hot_columns=None,
        remove_columns=None,
    ):
        """
        Splits the dataset into training and testing sets.
        """
        data = self.data.copy()
        random.seed(random_state)
        shuffled_indices = random.sample(range(len(data)), len(data))
        train_indices, test_indices = (
            shuffled_indices[: int(len(data) * (1 - test_size))],
            shuffled_indices[int(len(data) * (1 - test_size)) :],
        )
        X_train, y_train = self.prepare_data(
            index=train_indices,
            remove_columns=remove_columns,
            regression=regression,
            oversample=oversample,
            one_hot_columns=one_hot_columns,
            random_state=random_state,
            train=True,
        )
        X_test, y_test = self.prepare_data(
            index=test_indices,
            remove_columns=remove_columns,
            regression=regression,
            oversample=False,
            one_hot_columns=one_hot_columns,
            random_state=random_state,
            train=False,
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def convert_to_categorical(y_pred):
        """
        Converts the predicted numerical values to categorical values.
        """
        return [get_category(quality) for quality in y_pred]

    def prepare_data_inference(self, remove_columns=None, one_hot_columns=None):
        """
        Prepares the data for inference.
        """
        data = self.data.copy()
        # Remove the specified columns
        if remove_columns is not None:
            data.drop(columns=remove_columns, inplace=True)
        # One-hot encode the categorical variables
        if one_hot_columns is not None:
            data = pd.get_dummies(data, columns=one_hot_columns)
        # Scale the features
        if self.scaler is not None:
            return self.scaler.transform(data)
        else:
            return data

    def prepare_data(
        self,
        index=None,
        remove_columns=None,
        one_hot_columns=None,
        regression=False,
        oversample=False,
        train=False,
        random_state=42,
    ):
        """
        Prepares the data for training.
        """
        data = self.data.copy()
        # Set the index
        if index is not None:
            data = data.iloc[index]
        X, y = data.drop("Quality", axis=1), data["Quality"]
        # Oversample the training set
        if oversample:
            smote = SMOTE(random_state=random_state)
            X, y = smote.fit_resample(X, y)
        # Remove the specified columns
        if remove_columns is not None:
            X.drop(columns=remove_columns, inplace=True)
        # One-hot encode the categorical variables
        if one_hot_columns is not None:
            X = pd.get_dummies(X, columns=one_hot_columns)
        # Scale the features
        if self.scaler is not None:
            if train:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        # Convert the target variable to numerical values
        if regression:
            y = [price_map[quality] for quality in y]
        return X, y

    @staticmethod
    def compute_profit(y_pred):
        """
        Computes the profit based on the predicted quality.
        """
        sell = sum([price_map[quality] for quality in y_pred])
        packaging = len(y_pred) * 1
        return sell - packaging

    def save_scaler(self, path):
        """
        Saves the scaler to a file.
        """
        if self.scaler is not None:
            joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        """
        Loads the scaler from a file.
        """
        self.scaler = joblib.load(path)
