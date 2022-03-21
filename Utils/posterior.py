"""
Functions for handling MultiNest output files.

"""

import numpy as np
from scipy.stats import kurtosis
from pathlib import Path


def weighted_avg_and_std(values, weights, use_ulims=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=0, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, axis=0, weights=weights)

    if use_ulims:
        ulims = np.zeros_like(average)
        for i in range(ulims.size):
            ulims[i] = quantile(values[:, i], weights, 0.95)

    if use_ulims:
        return (average, np.sqrt(variance), ulims)
    else:
        return (average, np.sqrt(variance))

def quantile(data, weights, quantile):
    """
    Compute the quantile from a set of data and weights.

    Parameters
    ----------
    data : array-like
        Input (one-dimensional) array.
    weights : array-like
        Array of weights with shape matching `data`.
    quantile : float
        Quantile to compute in (0, 1].

    Returns
    -------
    q : float
        Quantile value.

    """
    sort_inds = np.argsort(data)
    d = data[sort_inds]
    w = weights[sort_inds]
    cdf_w = np.cumsum(w) / np.sum(w)
    q = np.interp(quantile, cdf_w, d)
    return q


class PosteriorCalculations(object):
    """
    Class containing functions for calculating and utilizing posteriors.

    Parameters
    ----------
    file_root : str
        Filename containing the sampler output.
    sampler : str
        Name of the sampler.  Defaults to 'multinest'.
    prefix : Path or str
        Path to the directory containing `file_root`.  Defaults to 'chains/'.
    nhistbins : int
        Number of histogram bins to use in the posteriors.  Defaults to 50.
    ulims : array
        Boolean array.  If True, posterior is interpreted as an upper limit.
        Otherwise, posterior is interpreted as a detection.

    """
    def __init__(self, file_root, sampler='multinest', prefix='chains',
                 nhistbins=50, ulims=None):
        self.sampler = sampler
        self.prefix = Path(prefix)
        self.file_root = file_root
        self.nhistbins = nhistbins
        self.ulims = ulims
        data, weights, posteriors, bin_edges = self.get_posteriors(
            self.prefix / self.file_root
        )
        self.data = data
        self.weights = weights
        self.ncoeffs = self.data.shape[1]
        self.posteriors = posteriors
        self.bin_edges = bin_edges
        self.calc_means_and_stddevs()
    
    def get_posteriors(self, fp):
        """
        Calculate posteriors from a numpy readable file.

        Parameters
        ----------
        fp : Path or str
            Path to sampler output file.

        Returns
        -------
        posteriors : array
            2D array containing histogrammed posteriors as probability density
            functions.
        bin_edges : array
            Bin edges from each histogrammed posterior.

        """
        data = np.genfromtxt(fp)
        if self.sampler == 'multinest':
            weights = data[:, 0]
            data = data[:, 2:]
            ncoeffs = data.shape[1]
        posteriors = np.zeros((ncoeffs, self.nhistbins))
        bin_edges = np.zeros((ncoeffs, self.nhistbins+1))

        for i_coeff in range(ncoeffs):
            amp, be = np.histogram(
                data[:, i_coeff], weights=weights,
                bins=self.nhistbins, density=True
            )
            posteriors[i_coeff] = amp
            bin_edges[i_coeff] = be

        return data, weights, posteriors, bin_edges
    
    def calc_means_and_stddevs(self):
        """
        Calculate the weighted mean and standard deviation of a posterior.

        This function doesn't return anything, but instead sets self.means
        and self.stddevs.

        """
        means, stddevs = weighted_avg_and_std(self.data, self.weights)
        self.means = means
        self.stddevs = stddevs

