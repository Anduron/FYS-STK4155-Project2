#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    print(states.shape)
    return states, energies


if __name__ == "__main__":

    from linearmodel import OLS, RidgeReg, LassoReg
    from sklearn.model_selection import train_test_split
    np.random.seed(42)

    L = 4     # system size
    N = 10  # number of points
    data, target = generate_1Ddata(L, N)
    ols = OLS(fit_intercept=False)
    X_train, X_test, y_train, y_test = ols.split_data(data, target, 0.96)

    coeff = ols.fit(X_train, y_train)
    ypredict = ols.predict(X_test)
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
    def morten():
        L = 4
        n = 10
        spins = np.random.choice([-1, 1], size=(n, L))
        J = 1.0
        energies = np.zeros(n)
        for i in range(n):
            energies[i] = - J * np.dot(spins[i], np.roll(spins[i], 1))
        X = np.zeros((n, L ** 2))
        for i in range(n):
            X[i] = np.outer(spins[i], spins[i]).ravel()
        y = energies
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.96)
        X_train_own = np.concatenate(
            (np.ones(len(X_train))[:, np.newaxis], X_train),
            axis=1)
        X_test_own = np.concatenate(
            (np.ones(len(X_test))[:, np.newaxis], X_test),
            axis=1)

    morten()
