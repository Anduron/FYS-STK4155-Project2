#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso


def frankeFunction(x, y):
    """
    Franke's function
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4


class LinearModel:
    """
    Linear model, superclass
    """

    def __init__(self, generate_design_matrix, normalize=False, add_intercept=False):
        self.generate_design_matrix = generate_design_matrix
        self.normalize = normalize
        self.intercept = add_intercept

    def design_matrix(self, x, normalize=False, intercept=False):
        X = self.generate_design_matrix(x)
        if normalize:
            X = self.normalize_design_matrix(X)

        if intercept:
            X = np.hstack((np.ones(N).reshape(-1, 1), X))

        return X

    def normalize_design_matrix(self, X, ignore_intercept=False):
        """
        Normalize design matrix
        """
        if ignore_intercept:
            Xp = X[:, 1:]
        else:
            Xp = X

        Xp_mean = np.mean(Xp, axis=0)
        Xp_std = np.std(Xp, axis=0)

        Xp_norm = (Xp - Xp_mean[np.newaxis, :]) / Xp_std[np.newaxis, :]

        if ignore_intercept:
            X = np.hstack((X[:, 0].reshape(-1, 1), Xp))
        else:
            X = Xp_norm

        return X

    def confidence_interval(self, p=.95):
        """
        Calculate CI
        """
        t = stats.t(df=self.N - self.eff_params).ppf(2 * p - 1)
        conf_intervals = [[self.b[i] - self.b_var[i] * t, self.b[i] + self.b_var[i] * t] for
                          i in range(self.params)]
        return conf_intervals

    def mse(self, x, y):
        """
        Calculate MSE
        """
        n = y.size
        _mse = 1 / n * np.sum((y - self.predict(x))**2)
        return _mse

    def rss(self, x, y):
        """
        Calculate RSS
        """
        _rss = np.sum((y - self.predict(x))**2)
        return _rss

    def r2(self, x, y):
        """
        Calculate R2
        """
        n = y.size
        y_ave = np.mean(y)
        _r2 = 1 - np.sum((y - self.predict(x))**2) / np.sum((y - y_ave)**2)
        return _r2


class OLS(LinearModel):
    """
    OLS, subclass of LinearModel
    """

    def __init__(self, generate_design_matrix):
        super().__init__(generate_design_matrix)

    def fit(self, x, y):
        """
        Fit data
        pinv - pseudoinverse (svd)
        """
        self.X = self.design_matrix(x)
        self.params = self.X.shape[1]
        self.eff_params = self.X.shape[1]
        self.N = self.X.shape[0]
        self.inv_cov_matrix = np.linalg.pinv(self.X.T @ self.X)
        self.b = self.inv_cov_matrix @ self.X.T @ y
        self.b_var = np.diag(self.inv_cov_matrix)
        self.b_var = self.N / (self.N - self.eff_params) * \
            self.mse(x, y) * self.b_var

    def predict(self, x):
        """
        Predict
        """
        pred = self.design_matrix(x) @ self.b
        return pred


class Ridge(LinearModel):
    """
    Ridge regression, subclass of LinearModel
    """

    def __init__(self, x, y, lamb, generate_design_matrix):
        super().__init__(generate_design_matrix, normalize=True)
        self.fit(x, y, lamb)

    def fit(self, x, y, lamb):
        """
        Fit
        pinv - pseudoinverse
        """
        self.X = self.design_matrix(x)

        self.inv_cov_matrix = np.linalg.pinv(
            X.T @ X + lamb * np.identity(self.X.shape[1]))

        self.params = self.X.shape[1] + 1
        self.eff_params = np.trace(X @ self.inv_cov_matrix @ X.T) + 1
        self.N = self.X.shape[0]
        self.b = np.zeros(self.params)
        self.b[0] = np.mean(y)
        self.b[1:] = self.inv_cov_matrix @ X.T @ y

        self.b_var = np.zeros(self.params)
        self.b_var[0] = 1 / self.N
        self.b_var[1:] = np.diag(self.inv_cov_matrix @
                                 X.T @ X @ self.inv_cov_matrix)
        self.b_var *= self.N / (self.N - self.eff_params) * self.mse(x, y)

    def predict(self, x):
        """
        Predict
        """
        pred = x @ self.b[1:] + self.b[0]
        return pred


class MyLasso(LinearModel):
    """
    Lasso regression, subclass of LinearModel
    """

    def fit(self, x, y, poly_deg, lamb):
        """
        fit
        """

        # build new model if never built before or if lamb/poly_deg changed
        if not hasattr(self, "lasso"):
            self.lasso = Lasso(
                alpha=lamb, fit_intercept=True, max_iter=1000000, warm_start=True)
        elif (not self.lamb == lamb) or (not self.poly_deg == poly_deg):
            self.lasso = Lasso(
                alpha=lamb, fit_intercept=True, max_iter=1000000, warm_start=True)

        self.lamb = lamb
        self.N = x.shape[0]
        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg, intercept=False)
        X = self.normalize_design_matrix(X)

        self.params += 1
        self.lasso.fit(X, y)
        self.b = np.zeros(self.params)
        self.b[0] = self.lasso.intercept_
        self.b[1:] = self.lasso.coef_

    def predict(self, x):
        """
        predict
        """
        X, P = self.design_matrix(x, self.poly_deg, intercept=False)
        X = self.normalize_design_matrix(X)
        pred = self.lasso.predict(X)
        return pred


def split_data(indicies, ratio=0.25):
    """
    Split data
    """
    n = len(indicies)
    test_set_size = int(ratio * n)
    rd.shuffle(indicies)
    test_idx = indicies[:test_set_size]
    train_idx = indicies[test_set_size:]
    return train_idx, test_idx


def kfold(indicies, k=5):
    """
    CV
    """
    n = len(indicies)
    rd.shuffle(indicies)
    N = ceil(n / k)
    indicies_split = []
    for i in range(k):
        a = i * N
        b = (i + 1) * N
        if b > n:
            b = n
        indicies_split.append(indicies[a:b])

    def folds(i):
        test_idx = indicies_split[i]
        train_idx = indicies_split[:i] + indicies_split[i + 1:]
        train_idx = [item for sublist in train_idx for item in sublist]
        return train_idx, test_idx

    return folds


def down_sample(terrain, N):
    """
    down sample
    """
    m, n = terrain.shape
    m_new, n_new = int(m / N), int(n / N)
    terrain_new = np.zeros((m_new, n_new))
    for i in range(m_new):
        for j in range(n_new):
            slice = terrain[N * i:N * (i + 1), N * j:N * (j + 1)]
            terrain_new[i, j] = np.mean(slice)
    return terrain_new


def plot_Franke():
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = frankeFunction(x, y)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(fig_path("franke_func.pdf"))
