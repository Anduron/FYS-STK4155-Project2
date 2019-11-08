#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LogisticRegression:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iter : int
        No of passes over the training set
    Attributes
    ----------
    coef_ : Estimated coefficients for the linear regression problem.
    intercept_ : Independent term in the linear model.
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.01, lmbda=0.0, n_iter=1000,
                 tol=1e-5, fit_intercept=True):
        self._eta = eta
        self._lmbda = lmbda
        self._n_iter = n_iter
        self._tol = tol
        self._fit_intercept = fit_intercept
        self.weights_ = None
        self.intercept_ = None

    def sigmoid(self, z):
        """
        Activation function; maps any real value between 0 and 1
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, method='GD'):
        """
        """
        #self.weights_ = np.zeros((X.shape[1], 1))
        self.weights_ = np.zeros(X.shape[1])
        self._method = method
        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            X = self.data.reshape(-1, 1)
        else:
            X = self.data
        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        if self._method == 'GD':
            self.GD()

        # set attributes
        if self._fit_intercept:
            self.intercept_ = self.weights_[0]
            self.weights_ = self.weights_[1:]
        else:
            self.intercept_ = 0

        return self.weights_

    def log_likelihood(self):
        Xbeta = self.data @ self.weights_
        return np.sum(self.target * Xbeta) - np.sum(np.log(1 + np.exp(Xbeta)))

    def gradient(self):
        """
        log-likelihood gradient
        """
        return self.data.T @ (self.target - self.predict_proba(self.data))

    def hessian(self):
        P = predict_proba(self.data)
        W = np.diag(P[0] * (1 - P[0]))
        return -self.X.T @ W @ self.X

    def GD(self):
        """
        Gradient descent
        """
        for _ in range(self._n_iter):
            self.weights_ -= self._eta * self.gradient()

    def predict_proba(self, X):
        """
        Returns the probability after passing through sigmoid
        """
        return self.sigmoid(X @ self.weights_)

    def predict(self, X):
        """
        Logistic regression model prediction
        """
        return 1 * (self.predict_proba(X) >= 0.5)


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"][:, :2]  # petal width
    y = (iris["target"] != 0) * 1  # 1 if Iris-Virginica, else 0
    print(y)

    logreg = LogisticRegression(fit_intercept=False)
    weights = logreg.fit(X, y)
    print(logreg.predict(X))
    print(logreg.log_likelihood())
    print(logreg.weights_)
    print(weights)
