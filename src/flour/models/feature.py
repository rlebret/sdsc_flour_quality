# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import pandas as pd
from abc import ABC, abstractmethod
import joblib


class FeatureModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def hyperparameters(self):
        pass

    def fit(self, X, y):
        """
        Fits the SVM classifier.

        parameters:
        - X: input features
        - y: target values
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the target values.

        parameters:
        - X: input features
        """
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the model to a file.

        parameters:
        - path: path to save the model
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Loads the model from a file.

        parameters:
        - path: path to load the model
        """
        self.model = joblib.load(path)

    def reset(self):
        """
        Resets the model.
        """
        self.model = clone(self.model)

    def cross_validate(
        self,
        data: pd.DataFrame,
        regression: bool = False,
        oversample: bool = False,
        n_splits=5,
        random_state=42,
    ):
        """
        Trains a machine learning model using the provided training
        data with cross-validation.

        Args:
            data: The training data.
            regression: Whether the task is a regression task.
            oversample: Whether to apply oversampling.
            n_splits: The number of splits for cross-validation.
            random_state: The random state for reproducibility.

        Returns:
            The ground truth labels and predictions.
        """
        # Define the number of folds (e.g., 5 or 10)
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )  # Adjust n_splits as needed

        # Lists to store results
        y_tests, y_preds = [], []
        for train_index, test_index in kfold.split(data.df(), y=data.target()):
            # Prepare the data for training
            X_train, y_train = data.prepare_data(
                index=train_index,
                regression=regression,
                oversample=oversample,
                train=True,
            )
            # Prepare the data for testing
            X_test, y_test = data.prepare_data(
                index=test_index,
                regression=regression,
                oversample=False,
                train=False,
            )
            # Train your model using the training data (X_train, y_train)
            self.model.fit(X_train, y_train)

            # Make predictions on the testing data (X_test)
            y_pred = self.model.predict(X_test)

            # Store the ground truth labels and predictions
            y_tests.append(y_test)
            y_preds.append(y_pred)
            # Clone the model to avoid overwriting
            self.reset()

        return y_tests, y_preds
