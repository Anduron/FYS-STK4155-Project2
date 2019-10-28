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
from sklearn.utils import resample

from linear_regression import *
from setup import *


def ols_stat(x, y, generate_design_matrix, generate_labels, p):
    model_ols = OLS(generate_design_matrix)
    model_ols.fit(x, y)
    N, L = x.shape
    labels = generate_labels(L)

    cmap = plt.get_cmap("Greens")

    # Dataframe for storing results
    df = pd.DataFrame(columns=['N', 'MSE', '$R^2$'])

    CI = model_ols.confidence_interval(p)
    mse = model_ols.mse(x, y)
    r2 = model_ols.r2(x, y)
    df = df.append({'N': N, 'MSE': mse,
                    '$R^2$': r2}, ignore_index=True)

    norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(CI))

    fig = plt.figure(figsize=(8, 6))
    plt.yticks(np.arange(model_ols.params), labels)
    plt.grid()

    for i in range(len(CI)):
        plt.plot(CI[i], (i, i), color=cmap(norm(i)))
        plt.plot(CI[i], (i, i), 'o', color=cmap(norm(i)))

    plt.gca().set_title(f'{p*100:.0f} % Confidence Interval')
    textstr = '\n'.join((
        f'$N = {N}$'))
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.gca().text(0.83, 0.95, f'$N = {N}$', transform=plt.gca().transAxes,
                   fontsize=14,  verticalalignment='top', bbox=props)
    fig.savefig(fig_path(f'conf_{N}.pdf'))

    # Render dataframe to a LaTeX tabular environment table and write to file
    pd.options.display.float_format = '{:,.3f}'.format
    df = df.apply(lambda x: x.astype(
        int) if np.allclose(x, x.astype(int)) else x)
    pd.options.display.latex.escape = False
    latex = df.to_latex(index=False, column_format='cccc')
    latex = latex.replace('\\toprule', '\\hline \\hline')
    latex = latex.replace('\\midrule', '\\hline \\hline')
    latex = latex.replace('\\bottomrule', '\\hline \\hline')

    with open(tab_path('ols_stat.tex'), 'w') as f:
        f.write(latex)


def OLS_split(x, y, generate_design_matrix, ratio=.25):
    """
    Perform data split and calculate training/testing MSE
    """
    N = x.shape[0]
    train_idx, test_idx = split_data(list(range(N)), ratio=ratio)
    model_ols = OLS(generate_design_matrix)
    model_ols.fit(x[train_idx], y[train_idx])
    mse_train = model_ols.mse(x[train_idx], y[train_idx])
    mse_test = model_ols.mse(x[test_idx], y[test_idx])

    df = pd.DataFrame(columns=['N', 'TrainMSE', 'TestMSE'])
    df = df.append({'N': N, 'TrainMSE': mse_train, 'TestMSE': mse_test},
                   ignore_index=True)
    print(df)


def OLS_CV(x, y, generate_design_matrix, k):
    """
    Calculate test MSE CV on OLS
    """
    model_ols = OLS(generate_design_matrix)
    N = x.shape[0]
    folds = kfold(list(range(N)), k)

    mse_train = 0
    mse_test = 0
    for j in range(k):
        train_idx, test_idx = folds(j)
        model_ols.fit(x[train_idx], y[train_idx])
        mse_train += model_ols.mse(x[train_idx], y[train_idx])
        mse_test += model_ols.rss(x[test_idx], y[test_idx])

    mse_train /= k
    mse_test /= N

    return mse_test


def ols_bias_variance(x, y, generate_design_matrix, n_bootstraps, ratio):
    n_boostraps = 10
    N = x.shape[0]
    model_ols = OLS(generate_design_matrix)
    # Hold out some test data that is never used in training.
    train_idx, test_idx = split_data(list(range(N)), ratio=ratio)

    x_train, y_train, x_test, y_test = x[train_idx], y[train_idx], x[test_idx], y[test_idx].reshape(
        -1, 1)

    # The following (m x n_bootstraps) matrix holds the column vectors y_pred
    # for each bootstrap iteration.
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)

        # Evaluate the new model on the same test data each time.
        model_ols.fit(x_, y_)
        y_pred[:, i] = model_ols.predict(x_test).ravel()

    # Note: Expectations and variances taken w.r.t. different training
    # data sets, hence the axis=1. Subsequent means are taken across the test data
    # set in order to obtain a total value, but before this we have error/bias/variance
    # calculated per data point in the test set.
    # Note 2: The use of keepdims=True is important in the calculation of bias as this
    # maintains the column vector form. Dropping this yields very unexpected results.
    print(f'y_test: {y_test.shape}, y_pred: {y_pred.shape}')
    print(y_pred)
    print(np.mean(y_pred, axis=1, keepdims=True))
    error = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)  # ??
    bias = np.mean(np.mean((y_pred - y_test), axis=1)**2)  # ??
    variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
    print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    print(f'{error} >= {bias} + {variance} = {bias + variance}')
