# Project 2: Classifying Phases of the 2D Ising Model with Logistic Regression and Deep Neural Networks

This repository contains programs, material and report for project 2 in FYS-STK4155 made in collaboration between [Kristian](https://github.com/KristianWold), [Tobias](https://github.com/vxkc) and [Nicolai](https://github.com/nicolossus).

In this project, we first use linear regression to determine the value of the coupling constant for the energy of the one-dimensional Ising model. Thereafter, we use two-dimensional data, but now computed at different temperatures, in order to classify the phase of the Ising model. Below the critical temperature, the system will be in a so-called ferromagnetic phase. Close to the critical temperature, the final magnetization becomes smaller and smaller in absolute value while above the critical temperature, the net magnetization is zero. This classification case, that is the two-dimensional Ising model, is studied using logistic regression and deep neural networks.

### Structure

The __[latex folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex)__ contains the LaTeX source for building the report, as well as __[figures](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/latex/figures)__ and __[tables](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/tables)__ generated in the analyses.

The __[notebooks folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/notebooks)__ contains Jupyter notebooks used in the analyses. For details, see the __[notebooks readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/notebooks/README.md)__.

The __[report folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/report)__ contains the report rendered to PDF from the LaTeX source.

The __[resources folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/resources)__ contains project resources such as supporting material, raw data to be analysed, etc.

The __[src folder](https://github.com/nicolossus/FYS-STK4155-Project2/tree/master/src)__ contains the source code, unit tests and benchmarks. For details, see the __[src readme](https://github.com/nicolossus/FYS-STK4155-Project2/blob/master/src/README.md)__.

### Documentation

[https://fys-stk4155-project2.readthedocs.io/en/latest/](https://fys-stk4155-project2.readthedocs.io/en/latest/)

### Usage

Execute unit tests of the implementations by `cd` into the project root folder and run `bash run.sh` in terminal.
