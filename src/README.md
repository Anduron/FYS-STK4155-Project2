Source code developed for the project.

Documentation: [https://fys-stk4155-project2.readthedocs.io/en/latest/](https://fys-stk4155-project2.readthedocs.io/en/latest/)

### Contents :shipit:

- [example.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/example.py): Example use. In more detail:
  - Generate noisy figure and save to latex folder
  - Generate table rendered to LaTeX environment and save to latex folder

- [isingmodel.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/isingmodel.py): Generate data for 1D Ising model.

- [linearmodel.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/linearmodel.py): Linear model with classes for
 - Ordinary Least Squares (OLS) Regression
 - Ridge Regression
 - Lasso Regression

- [logisticmodel.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/logisticmodel.py): Logistic model with class for
 - Logistic Regression

- [neuralnetwork.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/neuralnetwork.py): Perceptron neural network model.

- [ml_model_tools.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/ml_model_tools.py): Class with methods useful for machine learning models:
 - Split data set into training and test sets
 - Franke's bivariate test function

- [statistical_metrics.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/example.py): Class with methods for computing statistical metrics:
 - ``RSS`` - residual sum of squares
 - ``SST`` - total sum of squares
 - ``R2 score`` - coefficient of determination
 - ``MSE`` - mean squared error

- [test_models.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/test_models.py): Procedures for unit testing the models. See ``Usage`` below.

- [benchmark_ising_data.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/example.py): Performance benchmark on Ising model data generation.

- [project_tools.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/project_tools.py): Project structure specific tools:
 - Save figures to correct path
 - Generate tables rendered to LaTeX environment and save to correct path

- [benchmark_tools.py](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/benchmark_tools.py): Tools for benchmarking:
  - ``timer`` decorator: Return CPU time of a function call.


## Usage

See ``docstrings`` in source code or [docs](https://fys-stk4155-project2.readthedocs.io/en/latest/).

Execute unit tests of the implementation by `cd` into the project root folder and run `bash run.sh` in terminal.
