**Warning:** filenames that contains string 'example' will be deleted when `clean_examples.sh` is run.

# Project 2: Classifying Phases of the 2D Ising Model with Logistic Regression and Deep Neural Networks

This repository contains programs made for project X

In this project, we first use linear regression to determine the value of the coupling constant for the energy of the one-dimensional Ising model. Thereafter, we use two-dimensional data, but now computed at different temperatures, in order to classify the phase of the Ising model. Below the critical temperature, the system will be in a so-called ferromagnetic phase. Close to the critical temperature, the final magnetization becomes smaller and smaller in absolute value while above the critical temperature, the net magnetization is zero. This classification case, that is the two-dimensional Ising model, is studied using logistic regression and deep neural networks.

### Structure

The [latex folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex) contains the LaTeX source for building the report, as well as [figures](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex/figures) and [tables](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/tables) generated in the analyses.

The [notebooks folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/notebooks) contains Jupyter notebooks used in the analyses. For details, see the [notebooks readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/notebooks/README.md).

The [report folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/report) contains the report rendered to PDF from the LaTeX source.

The [resources folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/resources) contains project resources such as raw data to be analysed.

The [src folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/src) contains the source code. For details, see the [src readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/README.md).

The [test folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/test) contains procedures for unit testing and [benchmarking](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/test/benchmark) the source code developed for the project.

### Usage

To compile, test, benchmark and reproduce all results, `cd` into project and run `bash run.sh` in terminal.
