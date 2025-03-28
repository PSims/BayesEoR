""" Class and functions for reading/analyzing BayesEoR output files. """
import numpy as np
import scipy as sp
from pathlib import Path
import json
from jsonargparse import Namespace
from collections.abc import Iterable
import matplotlib.pyplot as plt


class DataContainer(object):
    """
    Class for reading and analyzing files output by BayesEoR.

    Parameters
    ----------
    dirnames : array-like of str
        Array-like of BayesEoR output directory names.
    dir_prefix : str, optional
        Prefix to append to `dirnames`. Defaults to None, i.e. the entries in
        `dirnames` are assumed to be valid paths to BayesEoR output
        directories.
    sampler : str, optional
        Case insensitive sampler name, e.g. 'MultiNest' or 'multinest'
        (default). The only currently supported sampler is MultiNest (default).
    store_samples : bool, optional
        If `store_samples` is True, store all samples as an attribute,
        `self.samples`.  Defaults to False.
    conf_intervals : float or list of float, optional
        Confidence intervals as percentages used for uncertainty calculations
        and errorbars on plots.  Defaults to [68, 95], i.e. compute 68% and 95%
        confidence intervals.
    calc_uplims : bool, optional
        If `calc_uplims` is True, calculate the upper limit of each k bin's
        posterior as the 95th percentile.  Defaults to False.
    uplim_quantile : float, optional
        Quantile in [0, 1].  Defaults to 0.95.  Only used if `calc_uplims`
        is True.
    uplim_inds : array-like, optional
        Array-like of True for non-detections and False for detections.  Can
        have shape ``(Nkbins,)``, where ``Nkbins`` is the number of spherically
        averaged k bins with power spectrum posteriors in the sampler output or
        shape ``(len(dirnames), Nkbins)``.  If ``Nkbins`` varies in each file,
        each entry in `uplims` must have ``len(uplims[i]) == Nkbins`` for that
        particular file.
    posterior_weighted : bool, optional
        If `posterior_weighted` is True, use the joint posterior probability as
        weights in the calculation of the individual power spectrum coefficient
        posteriors and the associated estimates, uncertainties, and upper
        limits.  Defaults to False.
    Nhistbins : int, optional
        Number of histogram bins for each k bin's posterior distribution.
    density : bool, optional
            If `density` is True, compute the posterior as a probability
            density function.  Defaults to False, i.e. plot counts.
    calc_kurtosis : bool, optional
        If `calc_kurtosis` is True, calculate the kurtosis of each k bin's
        posterior distribution.
    ps_kind : str, optional
        Case insensitive power spectrum kind in the sampler output file.  Can
        be 'ps' or 'dmps' for the power spectrum, :math:`P(k)`, or
        dimensionless power spectrum, :math:`\\Delta^2(k)`.  Defaults to 'dmps'.
    temp_unit : str, optional
        Either 'mK' or 'K'.  The temperature unit of the power spectrum.  The
        default output from BayesEoR is 'mK' (default).
    little_h_units : bool, optional
        If `little_h_units` is True, power spectrum samples are assumed to
        be in units of little h.  Defaults to False.  Currently, BayesEoR
        picks the Planck 2018 value of the Hubble constant but we plan to
        output power spectra in little h units in the future.  This kwarg
        has thus been added for future use.
    expected_ps : float or array-like, optional
        Expected power spectrum, :math:`P(k)`.  Can be a single float (for a
        flat :math:`P(k)`) or an array-like with length equal to the number of
        spherically averaged k bins.
    expected_dmps : float or array-like, optional
        Expected dimensionless power spectrum, :math:`\\Delta^2(k)`.  Can be a
        single float or an array-like with length equal to the number of
        spherically averaged k bins.
    labels : array-like of str, optional
        Array-like containing strings with shorthand labels for each directory
        in `dirnames`.  Used in figure legends for plotting functions.
        Defaults to None (no labels).
    
    """
    def __init__(
        self,
        dirnames,
        dir_prefix=None,
        sampler="multinest",
        store_samples=False,
        conf_intervals=[68, 95],
        calc_uplims=True,
        uplim_quantile=0.95,
        uplim_inds=None,
        posterior_weighted=True,
        Nhistbins=31,
        density=False,
        calc_kurtosis=False,
        ps_kind="dmps",
        temp_unit="mK",
        little_h_units=False,
        expected_ps=None,
        expected_dmps=None,
        labels=None
    ):
        self.Ndirs = len(dirnames)
        self.dirnames = dirnames
        if not isinstance(dir_prefix, Path) and dir_prefix is not None:
            dir_prefix = Path(dir_prefix)
        self.dir_prefix = dir_prefix
        self.sampler = sampler.lower()
        if self.sampler == 'multinest':
            self.sampler_filename = "data-.txt"
        self.labels = labels

        if not isinstance(conf_intervals, Iterable):
            conf_intervals = [conf_intervals]
        if uplim_inds is not None:
            self.calc_uplims = True
        else:
            self.calc_uplims = calc_uplims
        self.uplim_quantile = uplim_quantile
        self.uplim_inds = uplim_inds
        self.calc_kurtosis = calc_kurtosis

        # Units
        self.ps_kind = ps_kind.lower()
        self.temp_unit = temp_unit
        self.k_units = r"$h$ "*little_h_units + "Mpc$^{-1}$"
        if self.ps_kind == "ps":
            self.ps_label = r"$P(k)$"
        else:
            self.ps_label = r"$\Delta^2(k)$"
        self.ps_units = (
            fr"{self.temp_unit}$^2$"
            + r" $h^{-3}$"*little_h_units
            + r" Mpc$^3$"*(self.ps_kind == "ps")
        )

        # Load contents of output directories
        self.k_vals = []
        self.k_vals_bins = []
        self.version = []
        self.args = []
        self.posteriors = []
        self.posterior_bins = []
        self.avgs = []
        self.medians = []
        self.conf_intervals = []
        if self.calc_uplims:
            self.uplims = []
        if self.calc_kurtosis:
            self.kurtoses = []
        if store_samples:
            self.samples = []
            self.log_joint_posteriors = []
        for i_dir in range(self.Ndirs):
            path = Path(self.dirnames[i_dir])
            if self.dir_prefix is not None:
                path = self.dir_prefix / path
            k_vals = np.loadtxt(path / "k-vals.txt")
            k_vals_bins = np.loadtxt(path / "k-vals-bins.txt")
            with open(path / "version.txt", "r") as f:
                version = f.readlines()[0].strip("\n")
            with open(path / "args.json", "r") as f:
                args = Namespace(**json.load(f))
            self.k_vals.append(k_vals)
            self.k_vals_bins.append(k_vals_bins)
            self.version.append(version)
            self.args.append(args)

            # Get posteriors and power spectrum estimates
            # get_posterior_data returns the, optionally weighted, average,
            # standard deviation, 95th percentile (optional), and kurtosis
            # (optional) of each k bin's posterior distribution.
            out = self.get_posterior_data(
                path / self.sampler_filename,
                len(k_vals),
                posterior_weighted=posterior_weighted,
                conf_intervals=conf_intervals,
                uplim_quantile=self.uplim_quantile,
                Nhistbins=Nhistbins,
                density=density,
                log_priors=args.log_priors,
                return_samples=store_samples
            )
            self.posteriors.append(out[0])
            self.posterior_bins.append(out[1])
            self.avgs.append(out[2])
            self.medians.append(out[3])
            self.conf_intervals.append(out[4])
            if self.calc_uplims:
                self.uplims.append(out[5])
            if self.calc_kurtosis:
                self.kurtoses.append(out[6])
            if store_samples:
                self.samples.append(out[7][:, 2:])
                self.log_joint_posteriors.append(out[7][:, 0])
        
        if self.Ndirs > 1:
            self.k_vals_identical = np.all(np.diff(self.k_vals, axis=0) == 0)
        else:
            self.k_vals_identical = True

        # Store the expected (dimensionless) power spectra
        self.has_expected = (
            np.any([x is not None for x in [expected_ps, expected_dmps]])
        )
        if self.has_expected:
            self.calculate_expected_ps(
                expected_ps=expected_ps, expected_dmps=expected_dmps
            )
    
    def get_posterior_data(
        self,
        fp,
        Nkbins,
        posterior_weighted=False,
        conf_intervals=[68, 95],
        uplim_quantile=0.95,
        Nhistbins=31,
        density=False,
        log_priors=True,
        return_samples=False
    ):
        """
        Load sampler output and form posteriors for each k bin.

        Parameters
        ----------
        fp : str or Path
            Path to sampler output file.
        Nkbins : int
            Number of spherically averaged k bins.
        posterior_weighted : bool, optional
            If `posterior_weighted` is True, use the joint posterior
            probability as weights in the calculation of the individual power
            spectrum coefficient posteriors and the associated estimates,
            uncertainties, and upper limits.  Defaults to False.
        conf_intervals : float or list of floats, optional
            Confidence intervals as percentages.  Defaults to [68, 95], i.e.
            compute 68% and 95% confidence intervals.
        uplim_quantile : float, optional
            Quantile in [0, 1].  Defaults to 0.95.  Only used if
            `self.calc_uplims` is True.
        Nhistbins : int, optional
            Number of histogram bins for each k bin's posterior distribution.
            Defaults to 31.
        density : bool, optional
            If `density` is True, compute the posterior as a probability
            density function.  Defaults to False, i.e. plot counts.
        log_priors : bool, optional
            If `log_priors` is True (default), power spectrum samples are
            assumed to be in log10 units and are first linearized prior to
            calculating e.g. upper limits.  Otherwise, the power spectrum
            samples are assumed to be in linear units.
        return_samples : bool, optional
            If `return_samples` is True, return all samples. Defaults to False.
        
        Returns
        -------
        posteriors : ndarray
            Posterior distributions for each k bin with shape
            ``(Nkbins, Nhistbins)`` where `Nkbins` is the number of spherically
            averaged k bins in `fp`.
        avgs : ndarray
            Average of each k bin.
        medians : ndarray
            Median value of each k bin.
        ci_dict : dict
            Dictionary with the confidence interval(s) as key(s) and nested
            dictionaries for each confidence interval indexed by 'lo' and 'hi'
            for the low and high bounds of the confidence inverval,
            respectively.
        uplims : ndarray (returned only if `self.calc_uplims` is True)
            `uplim_quantile`-th quantile of each k bin.
        kurtoses : ndarray (returned only if `self.calc_kurtosis` is True)
            Kurtosis of each k bin's posterior.
        samples : ndarray (returned only if `return_samples` is True)
            Samples for each power power spectrum coefficient with shape
            ``(Nsamples, Nkbins + 2)``.

        """
        if not isinstance(fp, Path):
            fp = Path(fp)
        data = np.loadtxt(fp)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if log_priors:
            data[:, 2:] = 10**data[:, 2:]
        if self.sampler == 'multinest':
            # The relevant columns of the MultiNest output file 'data-.txt' are
            # 0 - joint posterior probability per iteration
            # 2: - these columns contain the power spectrum amplitude samples
            if posterior_weighted:
                weights = data[:, 0]
            else:
                weights = None
            avgs = np.average(data[:, 2:], axis=0, weights=weights)
            medians = self._weighted_quantiles(
                data[:, 2:], 0.5, weights=weights
            )
            ci_dict = {}
            for ci in conf_intervals:
                quantile = (ci/2 + 50) / 100
                lobounds = self._weighted_quantiles(
                    data[:, 2:], 1 - quantile, weights=weights
                )
                hibounds = self._weighted_quantiles(
                    data[:, 2:], quantile, weights=weights
                )
                ci_dict[ci] = {'lo': lobounds, 'hi': hibounds}
            if self.calc_uplims:
                uplims = self._weighted_quantiles(
                    data[:, 2:],
                    uplim_quantile,
                    weights=weights
                )

            posteriors = np.zeros((Nkbins, Nhistbins), dtype=float)
            bins = np.zeros((Nkbins, Nhistbins + 1), dtype=float)
            kurtoses = np.zeros(Nkbins, dtype=float)
            for i_k in range(data[0, 2:].size):
                bins_min = data[:, 2 + i_k].min()
                bins_max = data[:, 2 + i_k].max()
                if log_priors:
                    bins_min = np.log10(bins_min)
                    bins_max = np.log10(bins_max)
                bins[i_k] = np.logspace(bins_min, bins_max, Nhistbins + 1)
                posteriors[i_k], bins[i_k] = np.histogram(
                    data[:, 2 + i_k],
                    bins=bins[i_k],
                    density=density,
                    weights=weights
                )
                if self.calc_kurtosis:
                    kurtoses[i_k] = sp.stats.kurtosis(posteriors[i_k])
        else:
            print("'multinest' is currently the only supported sampler.")
            return
        
        return_vals = (posteriors, bins, avgs, medians, ci_dict)
        if self.calc_uplims:
            return_vals += (uplims,)
        else:
            return_vals += (None,)
        if self.calc_kurtosis:
            return_vals += (kurtoses,)
        else:
            return_vals += (None,)
        if return_samples:
            return_vals += (data,)
        else:
            return_vals += (None,)
        return return_vals

    def calculate_expected_ps(self, expected_ps=None, expected_dmps=None):
        """
        Calculated the expected power spectrum for each output file.

        Sets either `self.expected_ps` if `self.ps_kind` is 'ps' or
        `self.expected_dmps` if `self.ps_kind` is 'dmps'.

        Parameters
        ----------
        expected_ps : float or array-like, optional
            Expected power spectrum, :math:`P(k)`.  Can be a single float (for
            a flat :math:`P(k)`) or an array-like with length equal to the
            number of spherically averaged k bins.
        expected_dmps : float or array-like, optional
            Expected dimensionless power spectrum, :math:`\\Delta^2(k)`.  Can
            be a single float or an array-like with length equal to the number
            of spherically averaged k bins.

        """        
        if expected_ps is not None:
            # if hasattr(expected_ps, "__iter__"):
            if isinstance(expected_ps, Iterable):
                expected_ps = np.array(expected_ps)
                input_iterable = True
            else:
                input_iterable = False
        if expected_dmps is not None:
            # if hasattr(expected_dmps, "__iter__"):
            if isinstance(expected_dmps, Iterable):
                expected_dmps = np.array(expected_dmps)
                input_iterable = True
            else:
                input_iterable = False

        if expected_ps is not None and self.ps_kind == "ps":
            if not input_iterable:
                # Create an array if expected_ps is float
                expected_ps *= np.ones_like(self.k_vals[0])
            self.expected_ps = [expected_ps]
        elif expected_dmps is not None and self.ps_kind == "dmps":
            if not input_iterable:
                # Create an array if expected_dmps is float
                expected_dmps *= np.ones_like(self.k_vals[0])
            self.expected_dmps = [expected_dmps]
        elif expected_ps is not None and self.ps_kind == "dmps":
            # Convert from P(k) to \Delta^2(k)
            if self.k_vals_identical:
                expected_dmps = self._ps_to_dmps(expected_ps, self.k_vals[0])
                expected_dmps = [expected_dmps]
            else:
                expected_dmps = []
                for i in range(self.Ndirs):
                    expected_dmps.append(self._ps_to_dmps(
                            expected_ps, self.k_vals[i]
                    ))
            self.expected_dmps = expected_dmps
        elif expected_dmps is not None and self.ps_kind == "ps":
            # Convert from \Delta^2(k) to P(k)
            if self.k_vals_identical:
                expected_ps = self._dmps_to_ps(expected_dmps, self.k_vals[0])
                expected_ps = [expected_dmps]
            else:
                expected_ps = []
                for i in range(self.Ndirs):
                    expected_ps.append(self._dmps_to_ps(
                        expected_dmps, self.k_vals[i]
                    ))
            self.expected_ps = expected_ps
    
    def _ps_to_dmps(self, ps, ks):
        """
        Convert :math:`P(k)` to :math:`\\Delta^2(k)`.

        Parameters
        ----------
        ps : float or ndarray
            Float or ndarray of floats containing the power spectrum
            amplitude(s).  If an ndarray, must have the same shape as `ks`.
        ks : ndarray
            Array of k values.

        """
        return ps * ks**3 / (2 * np.pi**2)
    
    def _dmps_to_ps(self, dmps, ks):
        """
        Convert :math:`\\Delta^2(k)` to :math:`P(k)`.

        Parameters
        ----------
        dmps : float or ndarray
            Float or ndarray of floats containing the dimensionless power
            spectrum amplitude(s).  If an ndarray, must have the same shape as
            `ks`.
        ks : ndarray
            Array of k values.

        """
        return dmps * 2 * np.pi**2 / ks**3

    def plot_power_spectra(
        self,
        conf_interval=68,
        uplim_inds=None,
        plot_height=4.0,
        plot_width=7.0,
        hspace=0.05,
        height_ratios=[1, 0.5],
        x_offset=0,
        zorder_offset=0,
        labels=None,
        colors=None,
        cmap=None,
        marker="o",
        capsize=3,
        lw=3,
        ls_expected="--",
        plot_diff=False,
        plot_fracdiff=False,
        ylim=None,
        ylim_diff=[-1, 1],
        plot_priors=False,
        legend_loc="best",
        legend_ncols=1,
        figlegend=False,
        top=0.85,
        suptitle=None,
        fig=None,
        axs=None
    ):
        """
        Plot the power spectrum as the average with a confidence interval.

        Parameters
        ----------
        conf_interval : float, optional
            Confidence interval as a percentage to plot as the uncertainty.
            Defaults to 68.
        uplim_inds : array-like, optional
            Array-like of True for non-detections and False for detections.
            Can have shape ``(Nkbins,)``, where ``Nkbins`` is the number of
            spherically averaged k bins with power spectrum posteriors in the
            sampler output or shape ``(len(dirnames), Nkbins)``.  If ``Nkbins``
            varies in each file, each entry in `uplims` must have
            ``len(uplims[i]) == Nkbins`` for that particular file.  If
            `uplim_inds` is None (default), use `self.uplim_inds`.  Otherwise,
            use `uplim_inds` in place of `self.uplim_inds`.
        plot_height : float, optional
            Subplot height.  Defaults to 4.0.
        plot_width : float, optioanl
            Subplot width.  Defaults to 7.0.
        hspace : float, optional
            matplotlib gridspec height space.  Defaults to 0.05.  Only used if
            `plot_diff` or `plot_fracdiff` is True.
        height_ratios : array-like, optional
            matplotlib gridspec subplot height ratios.  Defaults to [1, 0.5],
            i.e. the top plot will be twice as tall as the bottom subplot.
            Only used if `plot_diff` or `plot_fracdiff` is True.
        x_offset : float, optional
            x-axis offset for plotting multiple results on a single subplot.
            If `x_offset` > 0, data points for different analyses will be
            offset along the x-axis to better distinguish overlapping data.
        zorder_offset : int, optional
            matplotlib zorder offset for plotting multiple results on a single
            subplot.  If `zorder_offset` > 0, data points for different
            analyses will be offset along the "z" axis (plot data points over
            or under each other).
        labels : array-like of str, optional
            Array-like of label strings for each analysis result.  If no labels
            are provided, checks for labels in `self.labels`.  If `self.labels`
            is not None and `labels` is not None, the labels in `labels` will
            be used instead of `self.labels`.  Must have length `self.Ndirs`.
            Defaults to None, i.e. use `self.labels` if not None otherwise use
            no labels.
        colors : array-like of str, optional
            Array-like of valid matplotlib color strings.  Must have length
            `self.Ndirs`.  If None (default), use the default matplotlib color
            sequence.
        cmap : callable matplotlib colormap instance, optional
            Callable matplotlib colormap instance, e.g.
            `matplotlib.pyplot.cm.viridis`.  If None (default), use the default
            matplotlib color sequence.
        marker : str, optional
            matplotlib marker string.  Defaults to 'o'.
        capsize : float, optional
            Errorbar cap size.  Defaults to 3.
        lw : float, optional
            matplotlib line width.  Defaults to 3.
        ls_expected : str, optional
            Any valid matplotlib line style string.  Used for plotting the
            expected power spectrum (used only if `self.expected_ps` or 
            `self.expected_dmps` is not None).  Defaults to '--'.
        plot_diff : bool, optional
            If `plot_diff` is True and `self.expected_ps` or
            `self.expected_dmps` is not None, plot the difference between each
            analysis' power spectrum and the expected power spectrum.  If
            both `plot_diff` and `plot_fracdiff` are True, `plot_diff` will be
            set to False and the fractional difference will be plotted instead.
            Defaults to False.
        plot_fracdiff : bool, optional
            If `plot_fracdiff` is True and `self.expected_ps` or
            `self.expected_dmps` is not None, plot the fractional difference
            between each analysis' power spectrum and the expected power
            spectrum.  If both `plot_diff` and `plot_fracdiff` are True,
            `plot_diff` will be set to False and the fractional difference will
            be plotted instead.  Defaults to False.
        ylim : array-like, optional
            matplotlib ylim for the power spectrum subplot.  Defaults to None
            (scales the y axis limits according to the data).
        ylim_diff : array-like, optional
            matplotlib ylim for the (fractional) difference subplot if
            `plot_diff` or `plot_fracdiff` is True.
        plot_priors : bool, optional
            If `plot_priors` is True, plot the prior bounds as shaded regions
            for each k bin.  Defaults to False.
        legend_loc : str, optional
            Any valid matplotlib legend locator string.  Used only if
            `figlegend` is False.  Defaults to 'best'.
        legend_ncols : int, optional
            Number of columns in the legend.  Defaults to 1 unless `figlegend`
            is True.  In the latter case, the default value is set to
            ``self.Ndirs + 1``.
        figlegend : bool, optional
            If `figlegend` is True, use a figure legend instead of an axes
            legend.  Defaults to False.
        top : float, optional
            Sets the top of the power spectrum subplot in figure fraction units
            (0, 1].  Defaults to 0.85.
        suptitle : str, optional
            Figure suptitle string.  Defaults to None.
        fig : Figure, optional
            matplotlib Figure instance.  Used internally when called by
            `self.plot_power_spectra_and_posteriors`.
        axs : Axes
            matplotlib Axes instance(s).  Used internally when called by
            `self.plot_power_spectra_and_posteriors`.

        """
        if plot_diff and plot_fracdiff:
            print(
                "Warning: `plot_diff` and `plot_fracdiff` cannot both be True."
                "  Setting `plot_diff` to False."
            )
            plot_diff = False
        
        if uplim_inds is None and self.uplim_inds is None:
            if self.k_vals_identical:
                uplim_inds = np.zeros(
                    (self.Ndirs, self.k_vals[0].size), dtype=bool
                )
            else:
                uplim_inds = []
                for i_dir in range(self.Ndirs):
                    uplim_inds.append(
                        np.zeros(self.k_vals[i_dir].shape, dtype=bool)
                    )

        external_call = np.all([x is not None for x in [fig, axs]])
        if external_call:
            # Being used by `self.plot_power_spectra_and_posteriors`
            subplots = len(axs) > 1
        else:
            subplots = plot_diff or plot_fracdiff and self.has_expected
            if subplots:
                fig_height = plot_height * np.sum(height_ratios) + 1
                fig_width = plot_width + 1
                figsize = (fig_width, fig_height)
                gridspec_kw = {
                    "hspace": hspace, "height_ratios": height_ratios
                }
                fig, axs = plt.subplots(
                    2, 1, figsize=figsize, sharex=True, gridspec_kw=gridspec_kw
                )
            else:
                fig_height = plot_height + 1
                fig_width = plot_width + 1
                figsize = (fig_width, fig_height)
                fig, ax = plt.subplots(figsize=figsize)
                axs = [ax]
        
        if cmap is not None:
            colors = cmap(np.linspace(0, 1, self.Ndirs))
        elif colors is None:
            colors = [f"C{i%10}" for i in range(self.Ndirs)]

        if self.Ndirs > 1 and x_offset == 0 and zorder_offset == 0:
            zorder_offset = 1

        ax = axs[0]
        ax.set_yscale("log")
        if subplots:
            ax_diff = axs[1]
        if self.has_expected:
            if self.ps_kind == "ps":
                expected = self.expected_ps
            else:
                expected = self.expected_dmps
            plot_expected = True
            i_exp = 0

        for i_dir in range(self.Ndirs):
            if self.has_expected:
                if self.k_vals_identical:
                    color = 'k'
                else:
                    color = colors[i_dir]
                if plot_expected:
                    ax.plot(
                        self.k_vals[i_dir] * (1 + x_offset*i_exp),
                        expected[i_exp],
                        color=color,
                        lw=lw,
                        ls=ls_expected,
                        label="Expected",
                        zorder=0 + zorder_offset*i_dir
                    )
            if labels is not None:
                label = labels[i_dir]
            elif self.labels is not None:
                label = self.labels[i_dir]
            else:
                label = None
            xs = self.k_vals[i_dir] * (1 + x_offset*i_dir)
            zorder = (10 + zorder_offset*i_dir)
            upl_inds = uplim_inds[i_dir]  # upper limits
            det_inds = np.logical_not(upl_inds)  # detections    
            ax.errorbar(  # plot upper limits
                xs[upl_inds],
                self.uplims[i_dir][upl_inds],
                yerr=self.uplims[i_dir][upl_inds].copy() * 2/3,
                uplims=True,
                color=colors[i_dir],
                marker=marker,
                capsize=capsize,
                lw=lw,
                ls="",
                zorder=zorder
            )
            yerr_lo = (
                self.medians[i_dir]
                - self.conf_intervals[i_dir][conf_interval]['lo']
            )
            yerr_hi = (
                self.conf_intervals[i_dir][conf_interval]['hi']
                - self.medians[i_dir]
            )
            yerr = np.array([yerr_lo, yerr_hi])
            ax.errorbar(  # plot detections
                xs[det_inds],
                self.medians[i_dir][det_inds],
                yerr=yerr[:, det_inds],
                color=colors[i_dir],
                marker=marker,
                capsize=capsize,
                lw=lw,
                ls="",
                label=label,
                zorder=zorder
            )

            if self.has_expected:
                if self.k_vals_identical:
                    color = 'k'
                else:
                    color = colors[i_dir]
                if subplots:
                    if plot_diff:
                        diff = self.medians[i_dir] - expected[i_exp]
                        diff_err = yerr.copy()
                        if np.any(uplim_inds[i_dir]) and self.calc_uplims:
                            diff[upl_inds] = (
                                self.uplims[i_dir][upl_inds]
                                - expected[i_exp][upl_inds]
                            )
                            diff_err[1, upl_inds] = np.abs(ylim_diff).max() / 2
                    else:
                        diff = self.medians[i_dir] / expected[i_exp] - 1
                        diff_err = yerr / expected[i_exp]
                        if np.any(uplim_inds[i_dir]) and self.calc_uplims:
                            diff[upl_inds] = (
                                self.uplims[i_dir][upl_inds]
                                / expected[i_exp][upl_inds]
                            )
                            diff[upl_inds] -= 1
                            diff_err[1, upl_inds] = np.abs(ylim_diff).max() / 2

                    ax_diff.errorbar(
                        xs[upl_inds],
                        diff[upl_inds],
                        yerr=diff_err[:, upl_inds],
                        uplims=True,
                        color=colors[i_dir],
                        marker=marker,
                        capsize=capsize,
                        lw=lw,
                        ls="",
                        zorder=zorder
                    )
                    ax_diff.errorbar(
                        xs[det_inds],
                        diff[det_inds],
                        yerr=diff_err[:, det_inds],
                        color=colors[i_dir],
                        marker=marker,
                        capsize=capsize,
                        lw=lw,
                        ls="",
                        zorder=zorder
                    )

                if self.k_vals_identical:
                    plot_expected = False
                else:
                    i_exp += 1

            if plot_priors:
                if i_dir == 0:
                    label_priors = "Priors"
                else:
                    label_priors = None
                priors_lo = np.array(self.args[i_dir].priors)[:, 0]
                priors_hi = np.array(self.args[i_dir].priors)[:, 1]
                if self.args[i_dir].log_priors:
                    priors_lo = 10**priors_lo
                    priors_hi = 10**priors_hi
                ax.stairs(
                    priors_hi,
                    self.k_vals_bins[i_dir],
                    baseline=priors_lo,
                    fill=True,
                    alpha=0.3,
                    label=label_priors,
                    zorder=0
                )

        if subplots:
            ax_diff.axhline(0, ls=ls_expected, color='k', lw=lw, zorder=0)
        
        # Axes labels
        xlabel = rf"$k$ [{self.k_units}]"
        if subplots:
            ax_diff.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel)
        ylabel = rf"{self.ps_label} [{self.ps_units}]"
        ax.set_ylabel(ylabel)
        if subplots:
            ylabel = "Difference"
            if plot_fracdiff:
                ylabel = "Fractional\n" + ylabel
            ax_diff.set_ylabel(ylabel)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        if subplots and ylim_diff is not None:
            ax_diff.set_ylim(ylim_diff)
        
        for ax_i in axs:
            ax_i.grid()
            ax_i.set_xscale("log")
        
        if not external_call:
            # Figure title
            if suptitle is not None:
                fig.suptitle(suptitle)

            # Add legend to subplot or figure
            if len(ax.get_legend_handles_labels()[0]) > 0:
                if legend_ncols == 1 and figlegend:
                    ncols = self.Ndirs + 1
                if not figlegend:
                    legend = ax.legend(loc=legend_loc, ncols=legend_ncols)
                    legend.set_zorder(100)
                elif figlegend:
                    if not subplots:
                        fig.tight_layout()
                    fig.subplots_adjust(top=top)
                    fig.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, top+0.1),
                        ncols=ncols,
                        frameon=False
                    )
        
            return fig
        else:
            return axs

    def plot_posteriors(
        self,
        plot_height=1.0,
        plot_width=7.0,
        hspace=0,
        colors=None,
        cmap=None,
        lw=3,
        ls_expected="--",
        log_y=False,
        ymin=1e-16,
        show_k_vals=True,
        plot_priors=False,
        suptitle=None,
        fig=None,
        axs=None
    ):
        """
        Plot power spectrum posteriors.

        Parameters
        ----------
        plot_height : float, optional
            Subplot height.  Defaults to 1.0.
        plot_width : float, optioanl
            Subplot width.  Defaults to 7.0.
        hspace : float, optional
            matplotlib gridspec height space.  Defaults to 0.05.  Only used if
            `plot_diff` or `plot_fracdiff` is True.
        colors : array-like of str, optional
            Array-like of valid matplotlib color strings.  Must have length
            `self.Ndirs`.  If None (default), use the default matplotlib color
            sequence.
        cmap : callable matplotlib colormap instance, optional
            Callable matplotlib colormap instance, e.g.
            `matplotlib.pyplot.cm.viridis`.  If None (default), use the default
            matplotlib color sequence.
        lw : float, optional
            matplotlib line width.  Defaults to 3.
        ls_expected : str, optional
            Any valid matplotlib line style string.  Used for plotting the
            expected power spectrum (used only if `self.expected_ps` or 
            `self.expected_dmps` is not None).  Defaults to '--'.
        log_y : bool, optional
            If `log_y` is True, plot the amplitudes of the posteriors in log10
            units.  Otherwise, plot the amplitudes of the posteriors in linear
            units (default).
        ymin : float, optional
            Minimum value for the y axis.  Used if `log_y` is True to avoid
            plotting very small y values.  Defaults to 1e-16.
        show_k_vals : bool, optional
            If `show_k_vals` is True (default), print the value of the k bin
            associated with each posterior in the upper left corner of each
            posterior subplot.
        plot_priors : bool, optional
            If `plot_priors` is True, plot the prior bounds as shaded regions
            for each k bin.  Defaults to False.
        suptitle : str, optional
            Figure suptitle string.  Defaults to None.
        fig : Figure, optional
            matplotlib Figure instance.  Used internally when called by
            `self.plot_power_spectra_and_posteriors`.
        axs : Axes
            matplotlib Axes instance(s).  Used internally when called by
            `self.plot_power_spectra_and_posteriors`.

        """
        external_call = np.all([x is not None for x in [fig, axs]])
        if not external_call:
            # Not being used by `self.plot_power_spectra_and_posteriors`
            # This plot is only useful for comparing power spectrum analyses
            # which share a spherical k binning scheme.
            Nkbins = self.k_vals[0].size
            fig_height = plot_height * (1 + hspace) * Nkbins + 1
            fig_width = plot_width + 1
            figsize = (fig_width, fig_height)
            gridspec_kw = {"hspace": hspace}
            fig, axs = plt.subplots(
                Nkbins, 1, figsize=figsize, sharex=True,
                gridspec_kw=gridspec_kw
            )

        if cmap is not None:
            colors = cmap(np.linspace(0, 1, self.Ndirs))
        elif colors is None:
            colors = [f"C{i%10}" for i in range(self.Ndirs)]

        if self.has_expected:
            if self.ps_kind == "ps":
                expected = self.expected_ps
            else:
                expected = self.expected_dmps

        for i_dir in range(self.Ndirs):
            for i_k, ax in enumerate(axs):
                ax.stairs(
                    self.posteriors[i_dir][i_k],
                    self.posterior_bins[i_dir][i_k],
                    color=colors[i_dir],
                    lw=lw
                )
                ax.set_yticks([])
                # ax.set_xlim([x_min, x_max])
                if i_dir == 0:
                    ax.grid()
                    if self.has_expected:
                        ax.axvline(
                            expected[i_dir][i_k],
                            lw=lw,
                            ls=ls_expected,
                            color="k",
                            zorder=0
                        )
                    ax.set_ylabel(fr"$\varphi_{i_k}$")
                    if show_k_vals:
                        k_val_str = (
                            fr"$k=${self.k_vals[0][i_k]:.1f} [{self.k_units}]"
                        )
                        ax.annotate(
                            k_val_str,
                            (0.02, 0.9),
                            xycoords="axes fraction",
                            ha="left",
                            va="top"
                        )
                    ax.set_xscale("log")
                    if log_y:
                        ax.set_ylim([ymin, ax.get_ylim()[1]])
                        ax.set_yscale("log")
                if plot_priors:
                    prior_lo = self.args[i_dir].priors[i_k][0]
                    prior_hi = self.args[i_dir].priors[i_k][1]
                    if self.args[i_dir].log_priors:
                        prior_lo = 10**prior_lo
                        prior_hi = 10**prior_hi
                    ax.axvspan(
                        prior_lo,
                        prior_hi,
                        color=colors[i_dir], 
                        alpha=0.3,
                        zorder=0
                    )
        ax.set_xlabel(fr"{self.ps_label} [{self.ps_units}]")

        if not external_call:
            ylabel = "Power Spectrum Coefficient Posterior Distributions"
            if log_y:
                ylabel = r"$\log_{10}$ " + ylabel
            fig.supylabel(ylabel)
            if suptitle is not None:
                fig.suptitle(suptitle)
            return fig
        else:
            return axs

    def plot_power_spectra_and_posteriors(
        self,
        conf_interval=68,
        uplim_inds=None,
        plot_height_ps=4.0,
        plot_width=7.0,
        hspace_ps=0.05,
        height_ratios_ps=[1, 0.5],
        x_offset=0,
        zorder_offset=0,
        labels=None,
        colors=None,
        cmap=None,
        marker="o",
        capsize=3,
        lw=3,
        ls_expected="--",
        plot_diff=False,
        plot_fracdiff=False,
        ylim_ps=None,
        ylim_diff_ps=[-1, 1],
        plot_priors=False,
        plot_height_post=1.0,
        hspace_post=0.01,
        log_y=False,
        ymin_post=1e-16,
        show_k_vals=True,
        legend_ncols=0,
        figlegend=True,
        top=0.875,
        right_ps=0.46,
        left_post=0.54,
        suptitle=None
    ):
        """
        Make a plot containing power spectra and posteriors as subplots.

        Calls `self.plot_power_spectra` and `self.plot_posteriors`.

        Parameters
        ----------
        conf_interval : float, optional
            Confidence interval as a percentage to plot as the uncertainty.
            Defaults to 68.
        uplim_inds : array-like, optional
            Array-like of True for non-detections and False for detections.
            Can have shape ``(Nkbins,)``, where ``Nkbins`` is the number of
            spherically averaged k bins with power spectrum posteriors in the
            sampler output or shape ``(len(dirnames), Nkbins)``.  If ``Nkbins``
            varies in each file, each entry in `uplims` must have
            ``len(uplims[i]) == Nkbins`` for that particular file.  If
            `uplim_inds` is None (default), use `self.uplim_inds`.  Otherwise,
            use `uplim_inds` in place of `self.uplim_inds`.
        plot_height_ps : float, optional
            Subplot height for the power spectra subplot(s).  Defaults to 4.0.
        plot_width : float, optioanl
            Subplot width for both the power spectra subplot(s) and the 
            posterior subplots.  Defaults to 7.0.
        hspace_ps : float, optional
            matplotlib gridspec height space for the power spectra subplot(s).
            Defaults to 0.05.  Only used if `plot_diff` or `plot_fracdiff` is
            True.
        height_ratios_ps : array-like, optional
            matplotlib gridspec subplot height ratios for the power spectra
            subplots.  Defaults to [1, 0.5], i.e. the top plot will be twice as
            tall as the bottom subplot.  Only used if `plot_diff` or
            `plot_fracdiff` is True.
        x_offset : float, optional
            x-axis offset for plotting multiple results on a single subplot.
            If `x_offset` > 0, data points for different analyses will be
            offset along the x-axis to better distinguish overlapping data.
        zorder_offset : int, optional
            matplotlib zorder offset for plotting multiple results on a single
            subplot.  If `zorder_offset` > 0, data points for different
            analyses will be offset along the "z" axis (plot data points over
            or under each other).
        labels : array-like of str, optional
            Array-like of label strings for each analysis result.  If no labels
            are provided, checks for labels in `self.labels`.  If `self.labels`
            is not None and `labels` is not None, the labels in `labels` will
            be used instead of `self.labels`.  Must have length `self.Ndirs`.
            Defaults to None, i.e. use `self.labels` if not None otherwise use
            no labels.
        colors : array-like of str, optional
            Array-like of valid matplotlib color strings.  Must have length
            `self.Ndirs`.  If None (default), use the default matplotlib color
            sequence.
        cmap : callable matplotlib colormap instance, optional
            Callable matplotlib colormap instance, e.g.
            `matplotlib.pyplot.cm.viridis`.  If None (default), use the default
            matplotlib color sequence.
        marker : str, optional
            matplotlib marker string.  Defaults to 'o'.
        capsize : float, optional
            Errorbar cap size.  Defaults to 3.
        lw : float, optional
            matplotlib line width.  Defaults to 3.
        ls_expected : str, optional
            Any valid matplotlib line style string.  Used for plotting the
            expected power spectrum (used only if `self.expected_ps` or 
            `self.expected_dmps` is not None).  Defaults to '--'.
        plot_diff : bool, optional
            If `plot_diff` is True and `self.expected_ps` or
            `self.expected_dmps` is not None, plot the difference between each
            analysis' power spectrum and the expected power spectrum.  If
            both `plot_diff` and `plot_fracdiff` are True, `plot_diff` will be
            set to False and the fractional difference will be plotted instead.
            Defaults to False.
        plot_fracdiff : bool, optional
            If `plot_fracdiff` is True and `self.expected_ps` or
            `self.expected_dmps` is not None, plot the fractional difference
            between each analysis' power spectrum and the expected power
            spectrum.  If both `plot_diff` and `plot_fracdiff` are True,
            `plot_diff` will be set to False and the fractional difference will
            be plotted instead.  Defaults to False.
        ylim_ps : array-like, optional
            matplotlib ylim for the power spectrum subplot.  Defaults to None
            (scales the y axis limits according to the data).
        ylim_diff_ps : array-like, optional
            matplotlib ylim for the (fractional) difference power spectrum
            subplot if `plot_diff` or `plot_fracdiff` is True.
        plot_priors : bool, optional
            If `plot_priors` is True, plot the prior bounds as shaded regions
            for each k bin.  Defaults to False.
        plot_height_post : float, optional
            Subplot height for the posterior subplots.  Defaults to 1.0.
        hspace_post : float, optional
            matplotlib gridspec height space for the posterior subplots.
            Defaults to 0.01.
        log_y : bool, optional
            If `log_y` is True, plot the amplitudes of the posteriors in log10
            units.  Otherwise, plot the amplitudes of the posteriors in linear
            units (default).
        ymin : float, optional
            Minimum value for the y axis of the posterior subplots.  Used if
            `log_y` is True to avoid plotting very small y values.  Defaults to
            1e-16.
        show_k_vals : bool, optional
            If `show_k_vals` is True (default), print the value of the k bin
            associated with each posterior in the upper left corner of each
            posterior subplot.
        legend_ncols : int, optional
            Number of columns in the legend.  Defaults to ``self.Ndirs + 1``.
        figlegend : bool, optional
            If `figlegend` is True (default), use a figure legend.  Otherwise,
            no legend is shown.
        top : float, optional
            Sets the top of the power spectrum and posterior subplots in figure
            fraction units (0, 1].  Defaults to 0.875.
        right_ps : float, optional
            Sets the right edge of the power spectrum subplot(s) in figure
            fraction units (0, 1].  Defaults to 0.46.
        left_post : float, optional
            Sets the left edge of the posterior subplot(s) in figure fraction
            units (0, 1].  Defaults to 0.54.
        suptitle : str, optional
            Figure suptitle string.  Defaults to None.

        """
        subplots_ps = np.any([plot_diff, plot_fracdiff])
        Nplots_ps = 1 + subplots_ps
        Nkbins = self.k_vals[0].size
        plots_height_ps = (
            plot_height_ps
            * (1 + height_ratios_ps[1]*subplots_ps)
            * (1 + hspace_ps*subplots_ps)
        )
        plots_height_post = plot_height_post * (1 + hspace_post) * Nkbins
        fig_height = np.max([plots_height_ps, plots_height_post])
        fig_width = plot_width * 2
        figsize = (fig_width, fig_height)
        fig = plt.figure(figsize=figsize)

        gridspec_kw_ps = {"hspace": hspace_ps, "right": right_ps}
        if subplots_ps:
            gridspec_kw_ps.update({"height_ratios": height_ratios_ps})
        if figlegend:
            gridspec_kw_ps.update({"top": top})
        gs_ps = fig.add_gridspec(Nplots_ps, 1, **gridspec_kw_ps)

        gridspec_kw_post = {"hspace": hspace_post, "left": left_post}
        if figlegend:
            gridspec_kw_post.update({"top": top})
        gs_post = fig.add_gridspec(Nkbins, 1, **gridspec_kw_post)

        # Create an empty axis to make a super ylabel for the posteriors
        temp_ax = fig.add_subplot(gs_post[:, :])
        temp_ax.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        for side in ["left", "right", "top", "bottom"]:
            temp_ax.spines[side].set_visible(False)
        ylabel = "Power Spectrum Coefficient Posterior Distributions"
        if log_y:
            ylabel = r"$\log_{10}$ " + ylabel
        temp_ax.set_ylabel(ylabel)

        axs_ps = gs_ps.subplots(sharex=True)
        # if not hasattr(axs_ps, "__iter__"):
        if not isinstance(axs_ps, Iterable):
            axs_ps = [axs_ps]
        axs_ps = self.plot_power_spectra(
            conf_interval=conf_interval,
            uplim_inds=uplim_inds,
            x_offset=x_offset,
            zorder_offset=zorder_offset,
            labels=labels,
            colors=colors,
            cmap=cmap,
            marker=marker,
            capsize=capsize,
            lw=lw,
            ls_expected=ls_expected,
            plot_diff=plot_diff,
            plot_fracdiff=plot_fracdiff,
            ylim=ylim_ps,
            ylim_diff=ylim_diff_ps,
            plot_priors=plot_priors,
            fig=fig,
            axs=axs_ps
        )
        
        axs_post = gs_post.subplots(sharex=True)
        axs_post = self.plot_posteriors(
            colors=colors,
            cmap=cmap,
            lw=lw,
            ls_expected=ls_expected,
            log_y=log_y,
            ymin=ymin_post,
            show_k_vals=show_k_vals,
            plot_priors=plot_priors,
            fig=fig,
            axs=axs_post
        )

        if figlegend:
            if suptitle is not None:
                fig.suptitle(suptitle)
            handles, labels_legend = axs_ps[0].get_legend_handles_labels()
            if len(handles) > 0:
                if legend_ncols == 0:
                    ncols = self.Ndirs + 1
                elif legend_ncols > 1:
                    ncols = legend_ncols
                fig.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, top+0.075),
                    ncols=ncols,
                    frameon=False
                )

        return fig

    def _weighted_quantiles(self, data, q, weights=None):
        """
        Compute a quantile from a set of data and optional weights.

        Parameters
        ----------
        data : array-like
            Input array.
        q : float
            Quantile in [0, 1].
        weights : array-like, optional
            Array of weights with shape matching `data` or
            ``(data.shape[0],)``.  If `weights` is None (default), uniformly
            weight the data.

        Returns
        -------
        quantile : float
            Quantile value.

        Notes
        -----
        This function should be replaced with `numpy.quantile` which, for
        ``numpy>=2.0``, now contains a `weights` kwarg.

        """
        if q < 0 or q > 1:
            raise ValueError("q must be in [0, 1]")
        if weights is None:
            weights = np.ones(data.shape[0])
        else:
            shapes_okay = (
                weights.shape == data.shape
                or weights.shape == data[:, 0].shape
            )
            assert shapes_okay, (
                "weights must have the same shape as data or data[:, 0]"
            )
        quantiles = np.ones(data.shape[1], dtype=float)
        for i in range(data.shape[1]):
            sort_inds = np.argsort(data[:, i])
            d = data[:, i][sort_inds]
            w = weights[sort_inds]
            cdf_w = np.cumsum(w) / np.sum(w)
            quantiles[i] = np.interp(q, cdf_w, d)
        return quantiles
