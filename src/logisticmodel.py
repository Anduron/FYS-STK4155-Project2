#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.special import expit
from scipy.special import xlogy
from ml_model_tools import MLModelTools


class LogisticRegression(MLModelTools):
    """
    Logistic Regression Using Optimization or Gradient Methods

    Parameters
    ----------
    eta : float
        Learning rate
    n_iter : int
        No of passes over the training set
    lmbda : float
        Regularization parameter
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
        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        if weights is not None:
            self.weights_ = weights
        else:
            if self._fit_intercept:
                self.weights_ = np.zeros(X.shape[1] + 1)
            else:
                self.weights_ = np.zeros(X.shape[1])

        self._method = method
        self._verbose = verbose

        # add bias if fit_intercept
        if self._fit_intercept:
            self.data = np.c_[np.ones(self.data.shape[0]), self.data]

        if self._method == 'GD':
            self.GD()
        elif self._method == 'NR':
            self.newton_raphson()

        return self.weights_

    def log_likelihood(self):
        Xbeta = self.data @ self.weights_
        return np.sum(self.target * Xbeta) - np.sum(np.log(1 + np.exp(Xbeta)))

    def cost(self):
        p = expit(self.data @ self.weights_)
        return np.sum(xlogy(self.target, p) + xlogy(1 - self.target, 1 - p))

    def gradient(self):
        """
        negative log-likelihood gradient
        """
        return - self.data.T @ (self.target - self.predict_proba(self.data)) + 2 * self._lmbda * self.weights_

    def hessian(self):
        """
        negative log-likelihood hessian
        """
        P = self.predict_proba(self.data)
        W = np.diag(P * (1 - P))
        return self.data.T @ W @ self.data + np.diag(2 * self._lmbda * np.ones(self.weights_.size))

    def GD(self):
        """
        Gradient descent
        """
        if self._verbose:
            print(f'Initial weights: {self.weights_}')
        for i in range(self._n_iter):
            weights_old = self.weights_
            self.weights_ = weights_old - self._eta * self.gradient()
            dL2 = np.linalg.norm(self.weights_ - weights_old)
            if self._verbose:
                print(f'Iteration no: {i}')
                print(f'New weights: {self.weights_}')
                print(f'L2 norm of weights: {np.linalg.norm(self.weights_)}')
                print(f'L2 change {dL2}')
                print(f'Log likelihood: {self.log_likelihood()}')
                print(f'Cost: {self.cost()}')
                print(
                    f'Accuracy: {self.accuracy(self.data[:,1:], self.target)}')
            if dL2 < self._tol or self.weights_[0] != self.weights_[0]:
                break

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
                print(f'New weight: {self.weights_}')
                print(f'L2 norm of weights: {np.linalg.norm(self.weights_)}')
                print(f'L2 change {dL2}')
                print(f'Log likelihood: {self.log_likelihood()}')
                print(f'Cost: {self.cost()}')
                print(
                    f'Accuracy: {self.accuracy(self.data[:,1:], self.target)}')
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
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
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
    logreg = LogisticRegression(lmbda=0, eta=0.0001, n_iter=1000)
    logreg.fit(X, y, verbose=True)
    print(logreg.weights_)
    print(logreg.predict(X))
    print(logreg.log_likelihood())
    print(logreg.accuracy(X, y))

    print("")
    print("NEWTON-RAPHSON\n")
    logreg = LogisticRegression(n_iter=10)
    weights = np.zeros(X.shape[1] + 1)
    logreg.fit(X, y, method='NR', verbose=True, weights=weights)
    print(logreg.predict(X))
    print(logreg.log_likelihood())
    print(logreg.weights_)
    print(logreg.accuracy(X, y))

    # weights = np.zeros(X.shape[1] + 1)
    # for i in range(10):
    #     weights = logreg.fit(X, y, method='NR', verbose=True, weights=weights)
    #     train_accuracy = logreg.accuracy(X, y)
    #     print('NR: %0.4f' % (train_accuracy))
