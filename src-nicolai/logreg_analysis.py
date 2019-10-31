#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ising import bias_variance, generate_1Ddata
from linearmodel import OLS, LassoReg, RidgeReg
from setup import fig_path

# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Import code from src
#sys.path.insert(0, '../src-nicolai/')

# set up Regression models
ols = OLS(fit_intercept=False)
ridge = RidgeReg(fit_intercept=False)
lasso = LassoReg(fit_intercept=False)

np.random.seed(42)
L = 40     # system size
N = 10000  # number of points
data, target = generate_1Ddata(L, N)
target = target + np.random.normal(0, 4.0, size=N)

X_train, X_test, y_train, y_test = ols.split_data(data, target, 0.96)

# define error lists
n = 10
# set regularisation strength values
lmbdas = np.logspace(-4, 5, n)

ols.fit(X_train, y_train)  # fit model
coef_ols = np.tile(ols.coef_, (n, 1))  # store weights

# use the coefficient of determination R^2 as the performance of prediction.
r2_train_ols = np.full(n, ols.r2(X_train, y_train))
r2_test_ols = np.full(n, ols.r2(X_test, y_test))

r2_train_ridge = np.zeros(n)
r2_test_ridge = np.zeros(n)

r2_train_lasso = np.zeros(n)
r2_test_lasso = np.zeros(n)

coefs_ridge = []
coefs_lasso = []

for i, lmbda in enumerate(lmbdas):
    # apply RIDGE regression
    ridge.set_penalty(lmbda)             # set regularisation parameter
    ridge.fit(X_train, y_train)          # fit model
    coefs_ridge.append(ridge.coef_)      # store weights
    # use the coefficient of determination R^2 as the performance of prediction
    r2_train_ridge[i] = ridge.r2(X_train, y_train)
    r2_test_ridge[i] = ridge.r2(X_test, y_test)

    # apply LASSO regression
    lasso.set_penalty(lmbda)             # set regularisation parameter
    lasso.fit(X_train, y_train)          # fit model
    coefs_lasso.append(lasso.coef_)      # store weights
    # use the coefficient of determination R^2 as the performance of prediction
    r2_train_lasso[i] = lasso.r2(X_train, y_train)
    r2_test_lasso[i] = lasso.r2(X_test, y_test)

    J_ols = np.array(ols.coef_).reshape((L, L))
    J_ridge = np.array(ridge.coef_).reshape((L, L))
    J_lasso = np.array(lasso.coef_).reshape((L, L))

    cmap_args = dict(vmin=-1., vmax=1., cmap='seismic')
    fig, axarr = plt.subplots(nrows=1, ncols=3)

    axarr[0].imshow(J_ols, **cmap_args)
    axarr[0].set_title('OLS \n Train$={:3f}$, Test$={:3f}$'.format(
        r2_train_ols[-1], r2_test_ols[-1]))

    axarr[1].imshow(J_ridge, **cmap_args)
    axarr[1].set_title('Ridge $\lambda={:4f}$\n Train$={:3f}$, Test$={:3f}$'.format(
        lmbda, r2_train_ridge[-1], r2_test_ridge[-1]))

    im = axarr[2].imshow(J_lasso, **cmap_args)
    axarr[2].set_title('Lasso $\lambda={:4f}$\n Train$={:3f}$, Test$={:3f}$'.format(
        lmbda, r2_train_lasso[-1], r2_test_lasso[-1]))

    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05, add_to_figure=True)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0 + 0.25, 0.25), fontsize=14)
    cbar.set_label('$J_{i,j}$', labelpad=15, y=0.5, fontsize=20, rotation=0)
    fig.subplots_adjust(right=2.0)
    # plt.show()


# print('Train ols', r2_train_ols)
# print('Test ols', r2_test_ols)
# print('Train ridge', r2_train_ridge)
# print('Test ridge', r2_test_ridge)
# print('Train lasso', r2_train_lasso)
# print('Test lasso', r2_test_lasso)
