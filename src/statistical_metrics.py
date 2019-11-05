#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class StatMetrics:
    """
    Compute Statistical Metrics

    Attributes
    ----------
    rss_ : float
        residual sum of squares
    sst_ : float
        total sum of squares
    r2_ : float
        coefficient of determination
    mse_ : float
        mean squared error

    Methods
    ----------
    rss(data, target)
        Calculate residual sum of squares of given input
    sst(target)
        Calculate total sum of squares of given input
    r2(data, target)
        Calculate coefficient of determination of given input
    mse(data, target)
        Calculate mean squared error of given input
    """

    def rss(self, data, target):
        """
        Calculates the residual sum of squares (RSS). It is a measure of the
        discrepancy between the data and an estimation model.

        Parameters
        ----------
        data : array, shape (n_samples)
            Data samples
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        C : float
            RSS value
        """

        return np.sum((target - self.predict(data))**2)

    def sst(self, target):
        """
        Calculates the total sum of squares (SST). It is defined as being the
        sum, over all observations, of the squared differences of each
        observation from the overall mean.

        Parameters
        ----------
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        C : float
            SST value
        """

        return np.sum((target - np.mean(target))**2)

    def r2(self, data, target):
        """
        Calculates the coefficient of determination (R2 score). It is the
        proportion of the variance in the dependent variable that is
        predictable from the independent variable(s).

        Parameters
        ----------
        data : array, shape (n_samples)
            Data samples
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        C : float
            R2 score value
        """

        return 1 - self.rss(data, target) / self.sst(target)

    def mse(self, data, target):
        """
        Calculates the mean squared error (MSE). It is a measure of the quality
        of an estimator—it is always non-negative, and values closer to zero
        are better.

        Parameters
        ----------
        data : array, shape (n_samples)
            Data samples
        target : array, shape (n_samples)
            Target samples

        Returns
        -------
        C : float
            MSE value
        """

        return np.mean((target - self.predict(data))**2)

    @property
    def rss_(self):
        """
        Calculated value of residual sum of squares (RSS) as class attribute
        """

        self._rss = np.sum((self.target - self.predict(self.data))**2)
        return self._rss

    @property
    def sst_(self):
        """
        Calculated value of total sum of squares (SST) as class attribute
        """

        self._sst = np.sum((self.target - np.mean(self.target))**2)
        return self._sst

    @property
    def r2_(self):
        """
        Calculated coefficient of determination (R2 score) as class attribute
        """

        self._r2 = 1 - self.rss_ / self.sst_
        return self._r2

    @property
    def mse_(self):
        """
        Calculated value of mean squared error (MSE) as class attribute
        """

        self._mse = np.mean((self.target - self.predict(self.data))**2)
        return self._mse
