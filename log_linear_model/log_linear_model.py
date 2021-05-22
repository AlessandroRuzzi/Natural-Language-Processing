import random
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import time

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
        self.feature_function = feature_function
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.iterations = iterations
        self.loss = loss
        self.gradient_loss = gradient_loss

    def gradient_descent(self, y: np.ndarray):
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
        self.gradient_value = self.gradient_loss(y,self.parameters,self.positive_features,self.negative_features)
        step_update = self.learning_rate * self.gradient_value
        self.parameters = np.subtract(self.parameters,step_update)

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
        self.parameters = np.random.rand(2*X.shape[1])
        self.positive_features = feature_function(X, np.ones(y.shape))
        self.negative_features = feature_function(X, np.zeros(y.shape))

        for epoch in range(self.iterations):
            prev_loss = self.loss(y,self.parameters,self.positive_features,self.negative_features)
            self.gradient_descent(y)
            curr_loss = self.loss(y,self.parameters,self.positive_features,self.negative_features)
            if self.verbose:
                abs_changes = np.abs(self.gradient_value)
                print(f"Current Iteration: {epoch}/{self.iterations} || Current loss: {curr_loss} || Change in loss: {np.abs(curr_loss-prev_loss)}"
                      f" || Absolute Largest value: {np.max(abs_changes)} Coefficient index: {np.argmax(abs_changes)}")

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
        self.positive_features = feature_function(X, np.ones(X.shape[0]))
        self.negative_features = feature_function(X, np.zeros(X.shape[0]))
        prediction = np.zeros(shape=(X.shape[0],2))
        for i in range(X.shape[0]):
            prediction[i][0] = probs(0,i, self.parameters,self.positive_features,self.negative_features)
            prediction[i][1] = probs(1,i, self.parameters, self.positive_features, self.negative_features)

        return np.argmax(prediction,axis=1)


def feature_function(X: np.ndarray, y: np.ndarray):
    features = np.zeros(shape=(X.shape[0], 2*X.shape[1]))
    for i in range(len(y)):
        if y[i] == 1:
            index = X.shape[1]
            features[i][index:] = X[i]
        else:
            index = X.shape[1]
            features[i][:index] = X[i]
    return features

def negative_log_likelihood(y: np.ndarray, parameters, positive_features,negative_features):
    nll = 0
    for i in range(len(y)):
            if y[i] ==1:
                numerator = np.dot(parameters, positive_features[i])
            else:
                numerator = np.dot(parameters, negative_features[i])

            denominator = - np.log(np.exp(np.dot(parameters, positive_features[i])) + np.exp(np.dot(parameters, negative_features[i])))
            nll += (numerator+denominator)

    return -nll


def probs(y: int,index, parameters, positive_features,negative_features):
    if y == 1:
        numerator = np.exp(np.dot(parameters, positive_features[index]))
    else:
        numerator = np.exp(np.dot(parameters, negative_features[index]))

    denominator = np.exp(np.dot(parameters, positive_features[index])) + np.exp(np.dot(parameters, negative_features[index]))

    return numerator/denominator


def gradient_negative_log_likelihood(y: np.ndarray, parameters, positive_features,negative_features):
    grad_nll = np.zeros(parameters.shape)
    for i in range(len(y)):
        if y[i] == 1:
            first_term_grad_nll = - positive_features[i]
        else:
            first_term_grad_nll = - negative_features[i]
        second_term_grad_nll = np.add(probs(1,i,parameters,positive_features,negative_features) * positive_features[i],
                                      probs(0,i,parameters,positive_features,negative_features) * negative_features[i])
        grad_nll = np.add(grad_nll, np.add(first_term_grad_nll, second_term_grad_nll))

    return grad_nll

def map_param(parameters : np.ndarray):
    new_parameters = np.zeros(shape=(int(parameters.shape[0]/2)))
    for i in range(int(len(parameters)/2)):
        new_parameters[i] = parameters[(int(parameters.shape[0]/2)) +i] - parameters[i]
    return new_parameters

# Set seeds to ensure reproducibility
np.random.seed(42)
random.seed(42)

lr_first = LogisticRegression()
learning_rate = 0.0001

llm_first = LogLinearModel(
    feature_function=feature_function,
    learning_rate=learning_rate,
    iterations=100,
    loss=negative_log_likelihood,
    gradient_loss=gradient_negative_log_likelihood,
    verbose=True
)

# First dataset
# Fit both `lr` and your `llm` on this dataset and compare
# the aspects described in the assignment PDF
X, y = make_classification(
    n_samples=100, random_state=42, n_informative=20, n_features=20, n_redundant=0
)

X_train_first, X_test_first, y_train_first, y_test_first = sklearn.model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


start_time = time.time()
llm_first.fit(X_train_first,y_train_first)
time_fit_llm_first = time.time() -start_time
start_time = time.time()
lr_first.fit(X_train_first,y_train_first)
time_fit_lr_first = time.time() -start_time

start_time = time.time()
llm_first.predict(X_train_first)
time_pred_llm_first = time.time() -start_time
start_time = time.time()
lr_first.predict(X_train_first)
time_pred_lr_first = time.time() -start_time

in_sample_llm_first = accuracy_score(y_true = y_train_first, y_pred = llm_first.predict(X_train_first))
out_sample_llm_first = accuracy_score(y_true = y_test_first, y_pred = llm_first.predict(X_test_first))
in_sample_lr_first = accuracy_score(y_true = y_train_first, y_pred = lr_first.predict(X_train_first))
out_sample_lr_first = accuracy_score(y_true = y_test_first, y_pred = lr_first.predict(X_test_first))

lr_second = LogisticRegression()
learning_rate = 0.0001

llm_second = LogLinearModel(
    feature_function=feature_function,
    learning_rate=learning_rate,
    iterations=100,
    loss=negative_log_likelihood,
    gradient_loss=gradient_negative_log_likelihood,
    verbose=True
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

X_train_second, X_test_second, y_train_second, y_test_second = sklearn.model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

start_time = time.time()
llm_second.fit(X_train_second,y_train_second)
time_fit_llm_second = time.time() -start_time
start_time = time.time()
lr_second.fit(X_train_second,y_train_second)
time_fit_lr_second = time.time() -start_time

start_time = time.time()
llm_second.predict(X_train_second)
time_pred_llm_second = time.time() -start_time
start_time = time.time()
lr_second.predict(X_train_second)
time_pred_lr_second = time.time() -start_time

in_sample_llm_second = accuracy_score(y_true = y_train_second, y_pred = llm_second.predict(X_train_second))
out_sample_llm_second = accuracy_score(y_true = y_test_second, y_pred = llm_second.predict(X_test_second))
in_sample_lr_second = accuracy_score(y_true = y_train_second, y_pred = lr_second.predict(X_train_second))
out_sample_lr_second = accuracy_score(y_true = y_test_second, y_pred = lr_second.predict(X_test_second))




lr_third = LogisticRegression()
learning_rate = 0.0001

llm_third = LogLinearModel(
    feature_function=feature_function,
    learning_rate=learning_rate,
    iterations=100,
    loss=negative_log_likelihood,
    gradient_loss=gradient_negative_log_likelihood,
    verbose=True
)
# Third dataset
# Fit both `lr` and your `llm` on this dataset and compare
# the aspects described in the assignment PDF
X, y = make_classification(
    n_samples=10000, random_state=42, n_informative=2, n_repeated=5
)

X_train_third, X_test_third, y_train_third, y_test_third = sklearn.model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

start_time = time.time()
llm_third.fit(X_train_third,y_train_third)
time_fit_llm_third = time.time() -start_time
start_time = time.time()
lr_third.fit(X_train_third,y_train_third)
time_fit_lr_third = time.time() -start_time

start_time = time.time()
llm_third.predict(X_train_third)
time_pred_llm_third = time.time() -start_time
start_time = time.time()
lr_third.predict(X_train_third)
time_pred_lr_third = time.time() -start_time

in_sample_llm_third = accuracy_score(y_true = y_train_third, y_pred = llm_third.predict(X_train_third))
out_sample_llm_third = accuracy_score(y_true = y_test_third, y_pred = llm_third.predict(X_test_third))
in_sample_lr_third = accuracy_score(y_true = y_train_third, y_pred = lr_third.predict(X_train_third))
out_sample_lr_third = accuracy_score(y_true = y_test_third, y_pred = lr_third.predict(X_test_third))


print(f"\n\nlog linear model train accuracy first dataset: {in_sample_llm_first}")
print(f"logistic regression train accuracy first dataset: {in_sample_lr_first}")
print(f"log linear model test accuracy first dataset: {out_sample_llm_first}")
print(f"logistic regression test accuracy first dataset: {out_sample_lr_first}")

print(f"\n\nlog linear model train accuracy second dataset: {in_sample_llm_second}")
print(f"logistic regression train accuracy second dataset: {in_sample_lr_second}")
print(f"log linear model test accuracy second dataset: {out_sample_llm_second}")
print(f"logistic regression test accuracy second dataset: {out_sample_lr_second}")

print(f"\n\nlog linear model train accuracy third dataset: {in_sample_llm_third}")
print(f"logistic regression train accuracy third dataset: {in_sample_lr_third}")
print(f"log linear model test accuracy third dataset: {out_sample_llm_third}")
print(f"logistic regression test accuracy third dataset: {out_sample_lr_third}")


data_dim = list((100,1000,10000))
values_in_sample_llm = list((in_sample_llm_first,in_sample_llm_second,in_sample_llm_third))
values_in_sample_lr = list((in_sample_lr_first,in_sample_lr_second,in_sample_lr_third))
values_out_sample_llm = list((out_sample_llm_first,out_sample_llm_second,out_sample_llm_third))
values_out_sample_lr = list((out_sample_lr_first,out_sample_lr_second,out_sample_lr_third))

values_time_fit_llm = list((time_fit_llm_first,time_fit_llm_second,time_fit_llm_third))
values_time_fit_lr = list((time_fit_lr_first,time_fit_lr_second,time_fit_lr_third))
values_time_pred_llm = list((time_pred_llm_first,time_pred_llm_second,time_pred_llm_third))
values_time_pred_lr = list((time_pred_lr_first,time_pred_lr_second,time_pred_lr_third))

plt.plot(data_dim, values_in_sample_llm,label = "Log Linear Model")
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.plot(data_dim, values_in_sample_lr,label = "Logistic Regression")
plt.legend()
plt.title('In-sample Accuracy')
plt.show()

plt.plot(data_dim, values_out_sample_llm,label = "Log Linear Model")
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.plot(data_dim, values_out_sample_lr,label = "Logistic Regression")
plt.legend()
plt.title('Out-of-sample Accuracy')
plt.show()

plt.plot(data_dim, values_time_fit_llm,label = "Log Linear Model")
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.plot(data_dim, values_time_fit_lr,label = "Logistic Regression")
plt.legend()
plt.title('Training Time')
plt.show()

plt.plot(data_dim, values_time_pred_llm,label = "Log Linear Model")
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.plot(data_dim, values_time_pred_lr,label = "Logistic Regression")
plt.legend()
plt.title('Prediction Time')
plt.show()


plt.scatter(list(range(map_param(llm_first.parameters).shape[0])), map_param(llm_first.parameters),label = "Log Linear Model")
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient values')
plt.scatter(list(range(lr_first.coef_.shape[1])), lr_first.coef_.T,label = "Logistic Regression")
plt.legend()
plt.title('Coefficients Values 100 samples')
plt.xticks(list(range(lr_first.coef_.shape[1])))
plt.show()

plt.scatter(list(range(map_param(llm_second.parameters).shape[0])), map_param(llm_second.parameters),label = "Log Linear Model")
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient values')
plt.scatter(list(range(lr_second.coef_.shape[1])), lr_second.coef_.T,label = "Logistic Regression")
plt.legend()
plt.title('Coefficients Values 1000 samples')
plt.xticks(list(range(lr_second.coef_.shape[1])))
plt.show()


plt.scatter(list(range(map_param(llm_third.parameters).shape[0])), map_param(llm_third.parameters),label = "Log Linear Model")
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient values')
plt.scatter(list(range(lr_third.coef_.shape[1])), lr_third.coef_.T,label = "Logistic Regression")
plt.legend()
plt.title('Coefficients Values 10000 samples')
plt.xticks(list(range(lr_third.coef_.shape[1])))
plt.show()











