import warnings
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14, 'figure.facecolor': 'w', 'lines.linewidth': 3.0
})
plt.style.use('seaborn-colorblind')

from BayesEoR.Utils import vector_is_hermitian
from BayesEoR.Utils.posterior import *
from BayesEoR.SimData import generate_data_and_noise_vector_instrumental
# from Likelihood.Likelihood import *  # initialize GPU comms

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    help='Path to numpy readable visibility data file in mK sr.'
)
parser.add_argument(
    '--array_dir',
    type=str,
    help='Directory containing all matrices and the MAP dictionary.'
)
parser.add_argument(
    '--filename',
    type=str,
    default='map-dict.npy',
    help='Filename for MAP dict in `array_dir`.  Defaults to \'map-dict.npy\'.'
)
parser.add_argument(
    '--file_root',
    type=str,
    help='Sampler output file root.'
)
parser.add_argument(
    '--expected_power',
    type=float,
    help='Expected power spectrum amplitude, i.e. P(k).  Currently only '
         'accepts a single floating point number representing a flat P(k).  '
         'Must have units of mK^2 Mpc^3.'
)
parser.add_argument(
    '--inv_lw_power',
    type=float,
    default=None,
    help='Prior on the inverse variance of the monopole, eta=0, and LSSM '
         'basis vectors.  Defaults to the value stored in `filename`.'
)
parser.add_argument(
    '--noise_seed',
    type=int,
    default=None,
    help='Random seed for noise generation.  Defaults to 742123.'
)
parser.add_argument(
    '--return_vars',
    action='store_true',
    help='If passed, return variables from functions (if running via ipython '
         'or in a notebook).'
)
parser.add_argument(
    '--print_timing',
    action='store_true',
    default=False,
    help='If passed, print GPU timing messages.'
)
args = parser.parse_args()

def plot_summary_plot(data_dict, k_vals, nhistbins=50, lw=4):
    """
    Plot the power spectrum, posteriors, matrix inversion accuracy, and MAP
    residuals histograms.

    """
    pc = data_dict['pc']
    inv_accuracy = data_dict['inv_accuracy']
    map_vis_res = data_dict['map_vis_res']
    noise = data_dict['n']
    keys_exp = ['inv_accuracy_exp', 'map_vis_exp_res', 'dmps_exp']
    plot_exp = np.any([key in data_dict for key in keys_exp])
    if plot_exp:
        inv_accuracy_exp = data_dict['inv_accuracy_exp']
        map_vis_exp_res = data_dict['map_vis_exp_res']
        dmps_exp = data_dict['dmps_exp']

    plot_height = 6
    plot_width = 8

    fig_width = plot_width * 2
    fig_height = plot_height * 3
    fig = plt.figure(figsize=(fig_width, fig_height))
    if plot_exp:
        top = 0.9
    else:
        top = 0.95
    buffer = 0.075 * top
    bottom = buffer
    left = buffer
    right = 1 - buffer
    
    plot_height_ratio = top * plot_height / fig_height
    gs_ps = fig.add_gridspec(
        1, 1,
        top=top,
        bottom=top - plot_height_ratio + buffer,
        left=left,
        right=0.5-0.65*buffer
    )
    gs_post = fig.add_gridspec(
        pc.ncoeffs, 1,
        hspace=0,
        top=top,
        bottom=top - plot_height_ratio + buffer,
        left=0.5+0.65*buffer,
        right=right
    )
    gs_inv = fig.add_gridspec(
        1, 2,
        top=gs_ps.bottom-buffer,
        bottom=gs_ps.bottom-buffer-plot_height_ratio+buffer, 
        left=left,
        right=right
    )
    gs_res = fig.add_gridspec(
        1, 2,
        top=gs_inv.bottom-buffer,
        bottom=bottom,
        left=left,
        right=right
    )
    if plot_exp:
        exp_fmt = 'k--'
        exp_c = 'k'
        exp_ls = '--'

    post_lbl_ax = fig.add_subplot(gs_post[:, :])
    post_lbl_ax.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    for side in ['left', 'right', 'top', 'bottom']:
        post_lbl_ax.spines[side].set_visible(False)
    post_lbl_ax.set_ylabel(
        'Power Spectrum Coefficient Posterior Distributions'
    )

    axs_all = []

    # Plot dimensionless power spectrum
    ax = fig.add_subplot(gs_ps[0])
    axs_all += [ax]
    ax.errorbar(
        k_vals, pc.means, yerr=pc.stddevs, marker='s', lw=lw, capsize=6,
        markeredgewidth=lw, label='Recovered', ls=''
    )
    if plot_exp:
        ax.plot(
            k_vals, np.log10(dmps_exp), exp_fmt, lw=lw, label='Expected'
        )
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax.set_ylabel(r'$\log_{10}\Delta^2(k)$ [mK$^2$]')

    # Plot posteriors
    axs = [fig.add_subplot(gs_post[i_coeff]) for i_coeff in range(pc.ncoeffs)]
    axs_all += axs
    for i_coeff, ax in enumerate(axs):
        ax.stairs(
            pc.posteriors[i_coeff], pc.bin_edges[i_coeff], ls='-', lw=lw
        )
        ax.set_yticks([])
        ax.set_ylabel(r'$\rho_{}$'.format(i_coeff))
        ax.annotate(
            r'$k$={:.3f} Mpc$^{{-1}}$'.format(k_vals[i_coeff]), (0.05, 0.95),
            xycoords='axes fraction', ha='left', va='top'
        )
    ax.set_xlabel(r'$\log_{10}\Delta^2(k)$ [mK$^2$]')
    min_x = np.min([ax.get_xlim()[0] for ax in axs])
    max_x = np.max([ax.get_xlim()[1] for ax in axs])
    for ax in axs:
        ax.set_xlim([min_x, max_x])

    # Plot inversion accuracy and MAP visibility residuals
    axs_inv = [fig.add_subplot(gs_inv[i]) for i in range(gs_inv.ncols)]
    axs_res = [fig.add_subplot(gs_res[i]) for i in range(gs_res.ncols)]
    axs_all += axs_inv + axs_res
    funcs = [np.real, np.imag]
    func_lbls = ['Re', 'Im']
    zip_obj = zip(axs_inv, axs_res, funcs, func_lbls)
    for i_col, (ax_inv, ax_res, func, func_lbl) in enumerate(zip_obj):
        _ = ax_inv.hist(
            func(inv_accuracy), lw=lw, histtype='step',
            bins=nhistbins, log=True, label='Recovered'
        )
        if plot_exp:
            _ = ax_inv.hist(
                func(inv_accuracy_exp), lw=lw, ls=exp_ls, color=exp_c,
                histtype='step', bins=nhistbins, log=True,
                label='Expected'
            )
        ax_inv.set_xlabel(
            f'{func_lbl}( '
            + r'$\bar{d} - \Sigma\Sigma^{-1}\bar{d}$'
            + ' )'
        )
        
        noise_std = noise.std()
        lbl = f'Noise\n$\sigma$={noise_std:.4e}'
        _ = ax_res.hist(
            func(noise) / noise_std, bins=nhistbins, color='k',
            alpha=0.5, label=lbl
        )
        res_std = map_vis_res.std()
        fe = (noise_std - res_std) / noise_std
        lbl = f'Recovered\n$\sigma$={res_std:.4e} ({fe*100:.2f}%)'
        _ = ax_res.hist(
            func(map_vis_res) / noise_std, bins=nhistbins, histtype='step',
            lw=lw, label=lbl
        )
        if plot_exp:
            res_exp_std = map_vis_exp_res.std()
            fe = (noise_std - res_exp_std) / noise_std
            lbl = f'Expected\n$\sigma$={res_exp_std:.4e} ({fe*100:.2f}%)'
            _ = ax_res.hist(
                func(map_vis_exp_res) / noise_std, bins=nhistbins,
                histtype='step', lw=lw, color=exp_c, ls=exp_ls, label=lbl
            )
        ax_res.set_xlabel(
            f'{func_lbl}( '
            + r'MAP Visibility Residuals'
            + r' ) [$\sigma_{\rm{noise}}$]'
        )
        if i_col == 0:
            ax_res.legend(loc='upper left')
    
    for ax in axs_all:
        ax.grid(which='both', alpha=0.5)

    fig.suptitle(pc.file_root)
    if plot_exp:
        handles, labels = axs_all[1+k_vals.size].get_legend_handles_labels()
        fig.legend(
            handles[::-1], labels[::-1], loc='upper center', ncol=2,
            frameon=False, bbox_to_anchor=(0.5, 1-0.75*buffer)
        )
    plt.show()

    return fig

def calc_SigmaI_dbar(pspp, dmps_coeffs):
    """
    Perform the Cholesky decomposition of Sigma to compute Sigma_inv * dbar.

    Parameters
    ----------
    pspp : PowerSpectrumPosteriorProbability
        Class containing variables and functions to perform the Cholesky
        decomposition.
    dmps_coeffs : array-like
        Dimensionless power spectrum amplitudes.

    Returns
    -------
    Sigma_copy : array
        Covariance matrix of dbar.
    map_uvetas : array
        MAP uveta-space amplitudes, i.e. SigmaI_dbar.

    """
    pspp.return_Sigma = True
    Sigma = pspp.calc_SigmaI_dbar_wrapper(
        dmps_coeffs, pspp.T_Ninv_T, pspp.dbar
    )
    pspp.return_Sigma = False
    Sigma_copy = Sigma.copy()
    map_uvetas = pspp.calc_SigmaI_dbar_wrapper(
        dmps_coeffs, pspp.T_Ninv_T, pspp.dbar
    )[0]

    return Sigma_copy, map_uvetas

def calculate_map_data(args):
    """
    Calculate maximum a posteriori data products.

    Parameters
    ----------
    args : Namespace
        Class containing parsed command line arguments.

    Returns
    -------
    map_dict : dictionary
        Dictionary containing all the necessary data products for computing
        MAP estimates.

    """
    data_path = Path(args.data_path)
    print(f'\nReading data from {data_path}', end='\n\n')
    data_dict = np.load(data_path, allow_pickle=True).item()
    vis = data_dict['data']

    print(f'Reading posteriors from {args.file_root}', end='\n\n')
    pc = PosteriorCalculations(args.file_root)

    array_dir = Path(args.array_dir)
    print(f'Reading MAP dict from {array_dir}', end='\n\n')
    map_dict = np.load(array_dir / args.filename, allow_pickle=True)
    if not isinstance(map_dict, dict):
        map_dict = map_dict.item()
    pspp = map_dict['pspp']
    bm = map_dict['bm']

    pspp.use_gpu = True
    pspp.initialize_gpu(print_msg=False)
    if not pspp.use_gpu:
        warnings.warn('GPU initialization failed, using CPU methods')
    pspp.T_Ninv_T = bm.read_data(str(array_dir / 'T_Ninv_T'), 'T_Ninv_T')
    pspp.Ninv = bm.read_data(str(array_dir / 'Ninv'), 'Ninv')
    sigma = 1 / np.sqrt(pspp.Ninv.real.diagonal()[0])
    T = bm.read_data(str(array_dir / 'T'), 'T')

    nbls = bm.uvw_array_m.shape[1]
    nt = bm.uvw_array_m.shape[0]
    if args.noise_seed:
        d, noise, bl_conj_pairs_map  = \
            generate_data_and_noise_vector_instrumental(
                sigma, vis, bm.nf, nt, bm.uvw_array_m[0], bm.bl_red_array[0],
                random_seed=args.noise_seed
            )
    else:
        _, _, bl_conj_pairs_map  = \
            generate_data_and_noise_vector_instrumental(
                sigma, vis, bm.nf, nt, bm.uvw_array_m[0], bm.bl_red_array[0],
                random_seed=args.noise_seed
        )
        noise = map_dict['n']
        d = vis + noise
    noise_std = noise.std()
    print(
        'signal is Hermitian:        ',
        vector_is_hermitian(
            vis, bl_conj_pairs_map, nt, bm.nf, nbls
        )
    )
    print(
        'signal + noise is Hermitian:',
        vector_is_hermitian(
            d, bl_conj_pairs_map, nt, bm.nf, nbls
        ),
        end='\n\n'
    )

    pspp.dbar = np.dot(T.conjugate().T, pspp.Ninv * d)

    if args.expected_power:
        dmps_exp = args.expected_power * pspp.k_vals**3 / (2 * np.pi**2)
    else:
        dmps_exp = None

    pspp.Print_debug = args.print_timing
    pspp.Print = args.print_timing
    if args.inv_lw_power is not None:
        pspp.inverse_LW_power = args.inv_lw_power

    print('-'*60 + '\n\nMAP Visibility Calculations', end='\n\n')

    dmps = 10**pc.means
    Sigma, map_uvetas = calc_SigmaI_dbar(pspp, dmps)
    map_vis = np.dot(T, map_uvetas)
    # If the MAP vis are a good fit to the input vis, then d - map_vis
    # should yield a distribution with a standard deviation comparable
    # to the noise
    map_vis_res = d - map_vis
    map_vis_res_std = map_vis_res.std()
    # Check the accuracy of the Sigma matrix inversion via the difference
    # dbar - Sigma * (Sigma_inv * dbar)
    ssidbar = np.dot(Sigma, map_uvetas)
    inv_accuracy = pspp.dbar - ssidbar

    print('Posterior (recovered):')
    print('noise.std()       =', noise_std)
    print('MAP Vis residuals =', map_vis_res_std)
    print(
        'Fractional error  =',
        (noise_std - map_vis_res_std) / noise_std,
        end='\n\n'
    )

    if dmps_exp is not None:
        Sigma_exp, map_uvetas_exp = calc_SigmaI_dbar(pspp, dmps_exp)
        map_vis_exp = np.dot(T, map_uvetas_exp)
        map_vis_exp_res = d - map_vis_exp
        map_vis_exp_res_std = map_vis_exp_res.std()
        ssidbar_exp = np.dot(Sigma_exp, map_uvetas_exp)
        inv_accuracy_exp = pspp.dbar - ssidbar_exp

        print('Posterior (expected):')
        print('noise.std()       =', noise_std)
        print('MAP Vis residuals =', map_vis_exp_res_std)
        print(
            'Fractional error  =',
            (noise_std - map_vis_exp_res_std) / noise_std,
            end='\n\n'
        )
    else:
        map_vis_exp = None
        inv_accuracy_exp = None
    
    print('-'*60, end='\n\n')

    data_dict = dict(
        s=vis,
        n=noise,
        d=d,
        pc=pc,
        pspp=pspp,
        bm=bm,
        dmps=dmps,
        inv_accuracy=inv_accuracy,
        map_uvetas=map_uvetas,
        map_vis=map_vis,
        map_vis_res=map_vis_res
    )
    if dmps_exp is not None:
        data_dict.update(dict(
            dmps_exp=dmps_exp,
            inv_accuracy_exp=inv_accuracy_exp,
            map_uvetas_exp=map_uvetas_exp,
            map_vis_exp=map_vis_exp,
            map_vis_exp_res=map_vis_exp_res
        ))
    fig = plot_summary_plot(data_dict, pspp.k_vals)

    if args.return_vars:
        return data_dict, fig
    else:
        return None, None


if __name__ == '__main__':
    data_dict, fig = calculate_map_data(args)
