# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from flour.models.feature import FeatureModel


class RandomForestFlour(FeatureModel):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        bootstrap=True,
    ):
        """
        Initializes the Random Forest classifier.

        parameters:
        - n_estimators: number of trees in the forest (default: 100)
        - max_depth: maximum depth of the tree (default: None)
        - min_samples_split: minimum number of samples required to split an internal
            node (default: 2)
        - min_samples_leaf: minimum number of samples required to be at a leaf
            node (default: 1)
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
        )

    @staticmethod
    def hyperparameters_randomized_search(X, y, n_iter=10):
        """
        Performs a randomized search for the hyperparameters.

        parameters:
        - X: input features
        - y: target values
        - n_iter: number of iterations
        """
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
            "max_features": ["auto", "sqrt"],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
        # Use random search to find the best hyperparameters
        rf_rand_search = RandomizedSearchCV(
            RandomForestClassifier(),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        # Fit the random search object to the data
        rf_rand_search.fit(X, y)
        return rf_rand_search.best_params_

    def hyperparameters(self):
        """
        Returns the hyperparameters of the model.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }
