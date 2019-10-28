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
    return [states, energies]


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


if __name__ == "__main__":

    from linearmodel import OLS, RidgeReg, LassoReg
    np.random.seed(42)

    L = 40     # system size
    N = 10000  # number of points
    data = generate_1Ddata(L, N)

    # define number of samples
    n_samples = 400
    # define train and test data sets
    X_train = data[0][:n_samples]
    # + np.random.normal(0,4.0,size=X_train.shape[0])
    Y_train = data[1][:n_samples]
    X_test = data[0][n_samples:3 * n_samples // 2]
    # + np.random.normal(0,4.0,size=X_test.shape[0])
    Y_test = data[1][n_samples:3 * n_samples // 2]

    model_ols = OLS(fit_intercept=False)      # Initialize model
    model_ols.fit(X_train, Y_train)
    # print(model_ols.mse)
    mse_train = model_ols.mse
    print(mse_train)
