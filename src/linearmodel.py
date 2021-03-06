#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Lasso as Lasso_
from sklearn.metrics import mean_squared_error, r2_score

from ml_model_tools import MLModelTools
from statistical_metrics import StatMetrics


class OLS(StatMetrics, MLModelTools):
    """
    Linear Model Using Ordinary Least Squares (OLS).

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    """

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array, shape = (n_samples) or shape = (n_samples, n_features)
            Training samples
        y : array, shape = (n_samples)
            Target values

        Returns
        -------
        Estimated coefficients for the linear regression problem : array, shape (n_features,)
        """

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

        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.pinv(xTx)
        xTy = np.dot(X.T, self.target)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Predicts the value after the model has been trained.

        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Test samples

        Returns
        -------
        Predicted values : array, shape (n_samples,)
        """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)


class Ridge(StatMetrics, MLModelTools):
    """
    Linear Model Using Ridge Regression.

    Parameters
    ----------
    lmbda : float, optional, default 1.0
        regularization (penalty) parameter; must be a positive float.
        Regularization improves the conditioning of the problem and reduces
        the variance of the estimates. Larger values specify stronger
        regularization.
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    """

    def __init__(self, lmbda=1.0, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array, shape = (n_samples) or shape = (n_samples, n_features)
            Training samples
        y : array, shape = (n_samples)
            Target values

        Returns
        -------
        Estimated coefficients for the linear regression problem : array, shape (n_features,)
        """

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

        xTx = np.dot(X.T, X)
        N = xTx.shape[0]
        inverse_xTx = np.linalg.pinv(xTx + self._lmbda * np.identity(N))
        xTy = np.dot(X.T, self.target)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Predicts the value after the model has been trained.

        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Test samples

        Returns
        -------
        Predicted values : array, shape (n_samples,)
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)

    def set_penalty(self, lmbda):
        """
        Set regularization parameter.

        Parameters
        ----------
        lmbda : float
            Value of regularization parameter

        Returns
        -------
        object : self
        """
        self._lmbda = lmbda
        return self


class Lasso(MLModelTools):
    """
    Linear Model Using LASSO Regression. Wraps sklearn's LASSO regression.

    Parameters
    ----------
    lmbda : float, optional, default 1.0
        regularization (penalty) parameter; must be a positive float.
        Regularization improves the conditioning of the problem and reduces
        the variance of the estimates. Larger values specify stronger
        regularization.
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    r2_ : float
        Coefficient of determination
    mse_ : float
        Mean squared error
    """

    def __init__(self, lmbda=1.0, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Training samples
        y : array, shape (n_samples)
            Target values

        Returns
        -------
        Estimated coefficients for the linear regression problem : array, shape (n_features,)
        """

        self.data = X
        self.target = y

        self.clf = Lasso_(alpha=self._lmbda, fit_intercept=self._fit_intercept)
        self.clf.fit(self.data, self.target)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = self.clf.intercept_
            self.coef_ = self.clf.coef_
        else:
            self.intercept_ = 0
            self.coef_ = self.clf.coef_

        return self.coef_

    def predict(self, X):
        """
        Predicts the value after the model has been trained.

        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Test samples

        Returns
        -------
        Predicted values : array, shape (n_samples,)
        """

        ypred = self.clf.predict(X)
        return ypred

    def set_penalty(self, lmbda):
        """
        Set regularization parameter.

        Parameters
        ----------
        lmbda : float
            Value of regularization parameter

        Returns
        -------
        self : object
        """

        self._lmbda = lmbda
        return self

    def mse(self, data, target):
        """
        Calculates the mean squared error (MSE). It is a measure of the quality
        of an estimator—it is always non-negative, and values closer to zero
        are better.

        Parameters
        ----------
        data : array, shape (n_samples)
            Data samples
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        MSE value : float
        """

        return mean_squared_error(target, self.predict(data))

    def r2(self, data, target):
        """
        Calculates the coefficient of determination (R2 score). It is the
        proportion of the variance in the dependent variable that is
        predictable from the independent variable(s).

        Parameters
        ----------
        data : array, shape (n_samples)
            Data samples
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        R2 score value : float
        """

        return r2_score(target, self.predict(data))

    @property
    def mse_(self):
        """
        Calculated value of mean squared error (MSE) as class attribute
        """

        self._mse = mean_squared_error(self.target, self.predict(self.data))
        return self._mse

    @property
    def r2_(self):
        """
        Calculated coefficient of determination (R2 score) as class attribute
        """

        self._r2 = r2_score(self.target, self.predict(self.data))
        return self._r2
