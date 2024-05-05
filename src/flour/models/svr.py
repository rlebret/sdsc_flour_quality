# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.model_selection import RandomizedSearchCV

from flour.models.feature import FeatureModel


class SVRFlour(FeatureModel):
    def __init__(
        self,
        kernel="rbf",
        C=1.0,
    ):
        """
        Initializes the SVM classifier.

        parameters:
        - kernel: kernel type (default: rbf)
        - C: regularization parameter (default: 1.0)
        - gamma: kernel coefficient (default: scale)
        """
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.model = SVR(kernel=kernel, C=C)

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
            "C": [0.1, 1, 10, 100, 1000],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
        }
        random_search = RandomizedSearchCV(
            SVR(), param_distributions=param_dist, n_iter=n_iter
        )
        random_search.fit(X, y)
        return random_search.best_params_

    def hyperparameters(self):
        """
        Returns the hyperparameters of the model.
        """
        return {"kernel": self.kernel, "C": self.C}
