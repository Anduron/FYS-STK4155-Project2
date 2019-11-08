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

    def __init__(self, eta=0.0001, lmbda=0.0, n_iter=1000,
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

    def fit(self, X, y, method='GD', verbose=False, weights=None):
        """
        """
        #self.weights_ = np.zeros((X.shape[1], 1))
        self.data = X
        self.target = y
        if weights is not None:
            self.weights_ = weights
        else:
            self.weights_ = np.zeros(X.shape[1])
        self._method = method
        self._verbose = verbose

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        # add bias if fit_intercept
        if self._fit_intercept:
            self.data = np.c_[np.ones(self.data.shape[0]), self.data]

        if self._method == 'GD':
            self.GD()
        elif self._method == 'NR':
            self.newton_raphson()

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
        return - self.data.T @ (self.target - self.predict_proba(self.data))

    def hessian(self):
        P = self.predict_proba(self.data)
        W = np.diag(P * (1 - P))
        return self.data.T @ W @ self.data

    def GD(self):
        """
        Gradient descent
        """
        for _ in range(self._n_iter):
            self.weights_ -= self._eta * self.gradient()

    def newton_raphson(self):
        """
        """
        if self._verbose:
            print(f'Initial weights: {self.weights_}')

        for i in range(self._n_iter):
            weights_old = self.weights_
            self.weights_ = weights_old - \
                np.linalg.pinv(self.hessian()) @ self.gradient()
            dL2 = np.linalg.norm(self.weights_ - weights_old)
            if self._verbose:
                print(f'Iteration no: {i}')
                # print(gradient.shape, hessian.shape)
                # print(f'gradient: {gradient}')
                # print(f'hessian: {hessian}')
                print(f'New beta: {self.weights_}')
                print(f'Log likelihood: {self.log_likelihood()}')
                print(f'L2 change {dL2}')
            if dL2 < self._tol or self.weights_[0] != self.weights_[0]:
                break

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

    def accuracy(self, X, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(X) >=
                             probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris["data"][:, :2]  # petal width
    y = (iris["target"] != 0) * 1  # 1 if Iris-Virginica, else 0

    print("GRADIENT DESCENT\n")
    logreg = LogisticRegression()
    logreg.fit(X, y)
    print(logreg.predict(X))
    print(logreg.log_likelihood())
    print(logreg.weights_)
    print(logreg.accuracy(X, y))

    print("")
    print("NEWTON-RAPHSON\n")
    logreg = LogisticRegression(n_iter=10)
    logreg.fit(X, y, method='NR', verbose=True)
    print(logreg.predict(X))
    print(logreg.log_likelihood())
    print(logreg.weights_)
    print(logreg.accuracy(X, y))
