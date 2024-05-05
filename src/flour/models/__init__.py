# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from flour.models.rf import RandomForestFlour
from flour.models.linear_regression import LinearRegressionFlour
from flour.models.svc import SVCFlour
from flour.models.svr import SVRFlour

__all__ = [
    "RandomForestFlour",
    "SVCFlour",
    "LinearRegressionFlour",
    "SVRFlour",
]
