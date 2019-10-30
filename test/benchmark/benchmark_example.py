import numpy as np

from benchmark_tools import timer


def generate_design_matrix_1(states):
    """
    Generate data for 1D Ising model that is optimized for linear regression
    """
    states = np.einsum('...i,...j->...ij', states, states)
    print(states.shape)
    shape = states.shape
    states = states.reshape((shape[0], shape[1] * shape[2]))
    print(states.shape)
    return states


def generate_design_matrix_2(states):
    """
    kool
    """
    i, j = np.triu_indices(states.shape[1])
    b = states[:, i] * states[:, j]
    print(b.shape)
    return states[:, i] * states[:, j]


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


states = np.random.choice([-1, 1], size=(10, 4))  # random Ising states
energies = ising_energies(states)  # calculate Ising energies


@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        generate_design_matrix_1(states)


@timer
def waste_some_time_2(num_times):
    for _ in range(num_times):
        generate_design_matrix_2(states)


if __name__ == "__main__":
    waste_some_time(1)
    waste_some_time_2(1)
