#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score

from linearmodel import OLS, Ridge, Lasso

# Create some data and target
np.random.seed(42)
n = 1000
x = np.random.rand(n, 1)
y = 5 * x * x + np.random.randn(n, 1)
X = x**2
x_lin = np.linspace(0, 1, 100)
X_lin = x_lin[:, np.newaxis]**2


def test_splitData():
    """
    Test if split_data method in MLModelTools splits data set into right ratio
    """
    ols = OLS()
    test_ratio = 0.2
    X_train, X_test, y_train, y_test = ols.split_data(X, y, test_ratio)
    assert abs(test_ratio - X_test.size / X.size) < 1e-8


def test_OLS_coef():
    """
    Test if home-made OLS model computes the same coefficient as sklearn's OLS
    """
    # Home-made model
    ols_own = OLS(fit_intercept=False)
    ols_own.fit(X, y)
    # Sklearn's model
    sklinreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    sklinreg.fit(X, y)

    assert abs(ols_own.coef_ - sklinreg.coef_) < 1e-8


def test_OLS_pred():
    """
    Test if home-made OLS model computes the same coefficient as sklearn's OLS
    """
    # Home-made model
    ols_own = OLS(fit_intercept=False)
    ols_own.fit(X, y)
    ypred_own = ols_own.predict(X_lin)
    # Sklearn's model
    sklinreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    sklinreg.fit(X, y)
    ypred_sklearn = sklinreg.predict(X_lin)

    assert abs(ypred_own[-1] - ypred_sklearn[-1]) < 1e-8


def test_mse():
    """
    Test if home-made mse method is implemented correctly in StatMetrics
    """
    # Home-made model
    ols_own = OLS(fit_intercept=False)
    ols_own.fit(X, y)
    # Sklearn's model
    sklinreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    sklinreg.fit(X, y)
    mse_sklearn = mean_squared_error(y, sklinreg.predict(X))

    assert abs(ols_own.mse_ - mse_sklearn) < 1e-8


def test_r2():
    """
    Test if home-made r2 method is implemented correctly in StatMetrics.
    (This also test both the rss and sst methods).
    """
    # Home-made model
    ols_own = OLS(fit_intercept=False)
    ols_own.fit(X, y)
    # Sklearn's model
    sklinreg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    sklinreg.fit(X, y)
    r2_sklearn = r2_score(y, sklinreg.predict(X))

    assert abs(ols_own.r2_ - r2_sklearn) < 1e-8


def test_Ridge_coef():
    """
    Test if home-made Ridge model computes the same coefficient as sklearn's
    """
    # Home-made model
    ridge_own = Ridge(lmbda=1.0, fit_intercept=False)
    ridge_own.fit(X, y)
    # Sklearn's model
    skridge = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=False)
    skridge.fit(X, y)

    assert abs(ridge_own.coef_ - skridge.coef_) < 1e-8


def test_Ridge_pred():
    """
    Test if home-made Ridge model computes the same coefficient as sklearn's
    """
    # Home-made model
    ridge_own = Ridge(lmbda=1.0, fit_intercept=False)
    ridge_own.fit(X, y)
    ypred_own = ridge_own.predict(X_lin)
    # Sklearn's model
    skridge = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=False)
    skridge.fit(X, y)
    ypred_sklearn = skridge.predict(X_lin)

    assert abs(ypred_own[-1] - ypred_sklearn[-1]) < 1e-8


def test_Lasso_coef():
    """
    Test if home-made Lasso model, which wraps sklearn's, is implemented
    correctly by checking that coefficents correspond
    """
    # Home-made model
    lasso_own = Lasso(lmbda=0.0, fit_intercept=False)
    lasso_own.fit(X, y)
    # Sklearn's model
    sklasso = sklearn.linear_model.Lasso(alpha=0.0, fit_intercept=False)
    sklasso.fit(X, y)

    assert abs(lasso_own.coef_ - sklasso.coef_) < 1e-8


def test_Lasso_pred():
    """
    Test if home-made Lasso model, which wraps sklearn's, is implemented
    correctly by checking that predictions correspond
    """
    # Home-made model
    lasso_own = Lasso(lmbda=1.0, fit_intercept=False)
    lasso_own.fit(X, y)
    ypred_own = lasso_own.predict(X_lin)
    # Sklearn's model
    sklasso = sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=False)
    sklasso.fit(X, y)
    ypred_sklearn = sklasso.predict(X_lin)

    assert abs(ypred_own[-1] - ypred_sklearn[-1]) < 1e-8
