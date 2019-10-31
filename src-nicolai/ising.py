#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils import resample


def ising_energies(states):
    """
    This function calculates the energies of the states in the nn Ising
    Hamiltonian
    """
    L = states.shape[1]
    J = np.zeros((L, L),)
    for i in range(L):
        J[i, (i + 1) % L] = -1.0  # interaction between nearest-neighbors
    # compute energies
    E = np.einsum('...i,ij,...j->...', states, J, states)
    return E


def generate_1Ddata(L=40, N=10000):
    """
    Generate data for 1D Ising model that is optimized for linear regression
    """
    states = np.random.choice([-1, 1], size=(N, L))  # random Ising states
    energies = ising_energies(states)  # calculate Ising energies
    # reshape Ising states into RL samples: S_iS_j --> X_p
    states = np.einsum('...i,...j->...ij', states, states)
    shape = states.shape
    states = states.reshape((shape[0], shape[1] * shape[2]))
    return states, energies


def bias_variance(model, data, target, n_bootstraps, ratio):
    # Hold out some test data that is never used in training.
    X_train, X_test, y_train, y_test = model.split_data(data, target, ratio)
    y_test = y_test.reshape(-1, 1)

    # The following (m x n_bootstraps) matrix holds the column vectors y_pred
    # for each bootstrap iteration.
    y_pred = np.empty((y_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X_, y_ = resample(X_train, y_train)

        # Evaluate the new model on the same test data each time.
        model.fit(X_, y_)
        y_pred[:, i] = model.predict(X_test).ravel()

    # Note: Expectations and variances taken w.r.t. different training
    # data sets, hence the axis=1. Subsequent means are taken across the test data
    # set in order to obtain a total value, but before this we have error/bias/variance
    # calculated per data point in the test set.
    # Note 2: The use of keepdims=True is important in the calculation of bias as this
    # maintains the column vector form. Dropping this yields very unexpected results.
    print(f'y_test: {y_test.shape}, y_pred: {y_pred.shape}')
    print(np.mean(y_pred, axis=1, keepdims=True))
    error = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias = np.mean(target - np.mean(y_pred, axis=1, keepdims=True))**2
    variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
    print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    print(f'{error} >= {bias} + {variance} = {bias + variance}')


if __name__ == "__main__":

    from linearmodel import OLS, RidgeReg, LassoReg
    from sklearn.model_selection import train_test_split
    # np.random.seed(42)

    L = 4     # system size
    N = 500  # number of points
    data, target = generate_1Ddata(L, N)
    target = target + np.random.normal(0, 4.0, size=N)
    ols = OLS(fit_intercept=False)
    print("split data")
    X_train, X_test, y_train, y_test = ols.split_data(data, target, 0.20)
    # print("fit")
    # coeff = ols.fit(X_train, y_train)
    # print("predict")
    # ypredict = ols.predict(X_test)
    print("bias-variance")
    bias_variance(ols, X_train, y_train, 20, .2)
    # print(coeff)
    # print(ypredict[-1])
    # print(ols.mse)
    # print(ols.r2)

    """
    model_ols = OLS(fit_intercept=False)      # Initialize model
    model_ols.fit(X_train, Y_train)
    # print(model_ols.mse)
    mse_train = model_ols.mse
    print(mse_train)
    """
