# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from sklearn.linear_model import LinearRegression
from flour.models.feature import FeatureModel


class LinearRegressionFlour(FeatureModel):
    def __init__(self):
        """
        Initializes the Linear Regression model.
        """
        super().__init__()
        self.model = LinearRegression()

    def get_coefficients(self):
        """
        Returns the coefficients of the model.
        """
        return self.model.coef_

    def hyperparameters(self):
        """
        Returns the hyperparameters of the model.
        """
        return {}
