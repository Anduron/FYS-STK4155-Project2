import numpy as np
import scipy.sparse as sp
import warnings
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from linear_regression import *
from setup import *
from analysis_ols import *


def ising_energies(states, L):
    """
    This function calculates the energies of the states in the nn Ising
    Hamiltonian
    """
    J = np.zeros((L, L),)
    for i in range(L):
        J[i, (i + 1) % L] -= 1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...', states, J, states)
    return E


def generate_design_matrix(states):
    i, j = np.triu_indices(states.shape[1])
    return states[:, i] * states[:, j]


def generate_labels(L):
    """
    Generate labels
    """
    return [f'$s_{i}s_{j}$' for i in range(1, L + 1) for j in range(i, L + 1)]


np.random.seed(14)

# Comment this to turn on warnings
warnings.filterwarnings('ignore')
# define Ising model aprams
# system size
L = 4
N = 20
# create 12 random Ising states
states = np.random.choice([-1, 1], size=(N, L))
# print(states)
# calculate Ising energies and add noise
energies = ising_energies(states, L) + np.random.normal(0, 4.0, size=N)

ols_stat(states, energies, generate_design_matrix, generate_labels, .95)
OLS_split(states, energies, generate_design_matrix, .25)
OLS_CV(states, energies, generate_design_matrix, 5)
ols_bias_variance(states, energies, generate_design_matrix, 100, .2)

# # for energy in energies:
# #    print(energy)
# print(energies)
# print(energies.shape)
