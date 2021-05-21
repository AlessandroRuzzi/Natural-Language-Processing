import random
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

class LogLinearModel:
    def __init__(
        self,
        feature_function: Callable,
        learning_rate: float,
        iterations: int,
        loss: Callable,
        gradient_loss: Callable,
        verbose: bool,
    ):
        """
        Parameters
        ---
        feature_function : Callable
            Feature function mapping from X x Y -> R^m
        learning_rate : float
            Learning rate parameter eta for gradient descent
        iterations : int
            Number of iterations to run gradient descent for during `fit`
        loss : Callable
            Loss function to be used by this LogLinearModel instance as
            a function of the parameters and the data X and y
        gradient_loss : Callable
            Closed form gradient of the `loss` function used for gradient descent as
            a function of the parameters and the data X and y
        verbose : bool
            Verbosity level of the class. If verbose == True,
            the class will print updates about the gradient
            descent steps during `fit`

        """
        # TODO
        pass

    def gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Performs one gradient descent step, and update parameters inplace.

        Parameters
        ---
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Binary target values

        Returns
        ---
        None

        """
        # TODO
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits LogLinearModel class using gradient descent.

        Parameters
        ---
        X : np.ndarray
            Input data matrix
        y : np.ndarray
            Binary target values

        Returns
        ---
        None

        """
        # TODO
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary target labels for input data `X`.

        Parameters
        ---
        X : np.ndarray
            Input data matrix

        Returns
        ---
        np.ndarray
            Predicted binary target labels

        """
        # TODO
        pass

# Set seeds to ensure reproducibility
np.random.seed(42)
random.seed(42)

lr = LogisticRegression()

llm = LogLinearModel(
    feature_function=feature_function,  # TODO
    learning_rate=learning_rate,  # Make sure that the model converges
                                  # with your chosen learning rate
    iterations=100,
    loss=negative_log_likelihood,  # TODO
    gradient_loss=gradient_negative_log_likelihood,  # TODO
)

# First dataset
# Fit both `lr` and your `llm` on this dataset and compare
# the aspects described in the assignment PDF
X, y = make_classification(
    n_samples=100, random_state=42, n_informative=20, n_features=20, n_redundant=0
)

# Second dataset
# Fit both `lr` and your `llm` on this dataset and compare
# the aspects described in the assignment PDF
X, y = make_classification(
    n_samples=1000,
    random_state=42,
    n_informative=20,
    n_redundant=10,
    n_features=35,
    n_repeated=5,
)

# Third dataset
# Fit both `lr` and your `llm` on this dataset and compare
# the aspects described in the assignment PDF
X, y = make_classification(
    n_samples=10000, random_state=42, n_informative=2, n_repeated=5
)