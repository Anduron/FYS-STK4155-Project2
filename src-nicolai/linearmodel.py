#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


class Metrics:
    """
    Compute statistical metrics
    """

    @property
    def rss(self):
        """
        return calculated value of rss (residual sum of squares)
        """
        self.rss_ = np.sum((self.target - self.predict(self.data))**2)
        return self.rss_

    @property
    def sst(self):
        """
        return calculated value of sst (total sum of squares)
        """
        self.sst_ = np.sum((self.target - np.mean(self.target))**2)
        return self.sst_

    @property
    def r2(self):
        """
        return calculated r2 score
        """
        self.r2_ = 1 - self.rss / self.sst
        return self.r2_

    @property
    def mse(self):
        """
        return calculated value of mse (mean squared error)
        """
        self.mse_ = np.mean((self.target - self.predict(self.data))**2)
        return self.mse_


class ModelTools:
    """
    Tools for ML models
    """

    def split_data(self, data, test_ratio=0.2):
        """
        ratio defaults to 20% test data, i.e. 80% training data
        caution: if not a seed is set, a new test set is generated by each run
        (which may cause problems since the ML algorithm will get to see the
        whole data set).
        return train_set, test_set
        """
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data[train_indices], data[test_indices]

    def design_matrix(self):
        """
        Redundant atm
        """
        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            X = self.data.reshape(-1, 1)
        else:
            X = self.data
        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X

    def generate_design_matrix(self, states):
        """
        kool
        """
        i, j = np.triu_indices(states.shape[1])
        return states[:, i] * states[:, j]

    def generate_labels(L):
        """
        Generate labels
        """
        l = [f'$s_{i}s_{j}$' for i in range(1, L + 1) for j in range(i, L + 1)]
        return l

    def design_matrix_interacting(self):
        """
        Under construction
        """
        pass

    def normalize_design_matrix(self):
        """
        Under construction
        """
        pass

    def frankeFunction(self, x, y):
        """
        Franke's bivariate test function - a widely used test function in
        interpolation and fitting problems.
        """
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) -
                              0.25 * ((9 * y - 2)**2))
        term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
        term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
        return term1 + term2 + term3 + term4

    def plot_Franke(self):
        """
        Plot Franke's function with x,y in [0, 1]
        """
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Make data.
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        x, y = np.meshgrid(x, y)

        z = self.frankeFunction(x, y)
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class OLS(Metrics, ModelTools):
    """
    Linear model class that fit and predict
    """

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit model coefficients

        Arguments:
        X: Data as 1D or 2D numpy array
        y: Target as 1D numpy array
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
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)


class RidgeReg(Metrics, ModelTools):
    """
    Linear model class that fit and predict

    Arguments (constructor):
    lmbda: regularization (penalty) parameter
    """

    def __init__(self, lmbda, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit model coefficients

        Arguments:
        X: Data as 1D or 2D numpy array
        y: Target as 1D numpy array
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
        """Output model prediction.

        Arguments:
        X: data as 1D or 2D numpy array
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)


class LassoReg(ModelTools):
    """
    Linear model class that fit and predict

    Arguments (constructor):
    lmbda: regularization (penalty) parameter
    """

    def __init__(self, lmbda, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit model coefficients, simple wrapper of sklearn Lasso

        Arguments:
        X: Data as 1D or 2D numpy array
        y: Target as 1D numpy array
        """

        self.data = X
        self.target = y

        self.clf = Lasso(alpha=self._lmbda, fit_intercept=self._fit_intercept)
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
        """Output model prediction.

        Arguments:
        X: data as 1D or 2D numpy array
        """
        ypred = self.clf.predict(X)
        return ypred

    @property
    def mse(self):
        self.mse_ = mean_squared_error(self.target, self.predict(self.data))
        return self.mse_

    @property
    def r2(self):
        self.r2_ = r2_score(self.target, self.predict(self.data))
        return self.r2_


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    import warnings
    # Comment this to turn on warnings
    warnings.filterwarnings('ignore')

    # Create some data and target
    np.random.seed(42)
    n = 1000
    x = np.random.rand(n, 1)
    y = 5 * x * x + np.random.randn(n, 1)
    X = x**2
    x_lin = np.linspace(0, 1, 100)
    X_lin = x_lin[:, np.newaxis]**2

    """
    TEST OLS
    """
    # MY MODEL
    ols = OLS(fit_intercept=False)
    coef = ols.fit(X, y)
    ypred = ols.predict(X_lin)

    train_set, test_set = ols.split_data(X)
    print("Train data", train_set.size)
    print("Test data", test_set.size)
    print("Test ratio", test_set.size / X.size)

    print("My OLS")
    print(f"coef: {ols.coef_}")
    print(f"Predict: {ypred[-1]}")
    print(f"MSE: {ols.mse}")
    print(f"R2 score: {ols.r2}")

    # SKLEARN
    sklinreg = LinearRegression(fit_intercept=False)
    coeff = sklinreg.fit(X, y)
    ypredict = sklinreg.predict(X_lin)
    print("")
    print("scikit OLS")
    print(f"coef: {sklinreg.coef_}")
    print(f"Predict: {ypredict[-1]}")
    print(f"Mean squared error: {mean_squared_error(y, sklinreg.predict(X))}")
    print(f"R2 score: {r2_score(y, sklinreg.predict(X))}")

    """
    TEST RIDGE
    """
    # MY MODEL
    ridge = RidgeReg(lmbda=1.0, fit_intercept=False)
    coef = ridge.fit(X, y)
    ypred = ridge.predict(X_lin)
    print("")
    print("My Ridge")
    print(f"coef: {ridge.coef_}")
    print(f"Predict: {ypred[-1]}")
    print(f"MSE: {ridge.mse}")
    print(f"R2 score: {ridge.r2}")

    # SKLEARN
    skridge = Ridge(alpha=1.0, fit_intercept=False)
    skridge.fit(X, y)
    ypredict = skridge.predict(X_lin)
    print("")
    print("scikit Ridge")
    print(f"coef: {skridge.coef_}")
    print(f"Predict: {ypredict[-1]}")
    print(f"Mean squared error: {mean_squared_error(y, skridge.predict(X))}")
    print(f"R2 score: {r2_score(y, skridge.predict(X))}")

    """
    TEST LASSO
    """
    # MY MODEL
    lasso = LassoReg(lmbda=0.0, fit_intercept=False)
    coef = lasso.fit(X, y)
    ypred = lasso.predict(X_lin)
    print("")
    print("My Lasso")
    print(f"coef: {lasso.coef_}")
    print(f"Predict: {ypred[-1]}")
    print(f"MSE: {lasso.mse}")
    print(f"R2 score: {lasso.r2}")

    # SKLEARN
    sklasso = Lasso(alpha=0.0, fit_intercept=False)
    sklasso.fit(X, y)
    ypredict = sklasso.predict(X_lin)
    print("")
    print("scikit Lasso")
    print(f"coef: {sklasso.coef_}")
    print(f"Predict: {ypredict[-1]}")
    print(f"Mean squared error: {mean_squared_error(y, sklasso.predict(X))}")
    print(f"R2 score: {r2_score(y, sklasso.predict(X))}")
