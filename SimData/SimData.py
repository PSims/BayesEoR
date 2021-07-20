import numpy as np
from numpy import * # don't know if this is necessary
from subprocess import os

import BayesEoR.Params.params as p


def generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
        nu, nv, nf, neta, ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc, ps_box_size_para_Mpc):
    # Rename this function? This is the default funciton
    # Generate k_cube pixel coordinates
    z, y, x = np.mgrid[-(nf//2) : (nf//2),
                       -(nv//2) : (nv//2)+1,
                       -(nu//2) : (nu//2)+1]

    # Setup k-space arrays
    deltakx = 2.*np.pi / ps_box_size_ra_Mpc
    deltaky = 2.*np.pi / ps_box_size_dec_Mpc
    deltakz = 2.*np.pi / ps_box_size_para_Mpc
    k_z = z * deltakz
    k_y = y * deltaky
    k_x = x * deltakx
    mod_k_physical = (k_z**2. + k_y**2. + k_x**2.)**0.5

    return mod_k_physical, k_x, k_y, k_z, x, y, z


def generate_data_and_noise_vector_instrumental(
        sigma, s, nu, nv, nf, neta, nq, nt,
        uvw_array_meters, bl_redundancy_array, **kwargs):
    # Need to rename this function, the name is too long
    # ===== Defaults =====
    default_random_seed = ''

    # ===== Inputs =====
    random_seed = kwargs.pop('random_seed', default_random_seed)

    if sigma == 0.0:
        complex_noise_hermitian = np.zeros(len(s)) + 0.0j
        d = s + complex_noise_hermitian.flatten()
        return d, complex_noise_hermitian.flatten()

    nbls = len(uvw_array_meters)
    ndata = nbls * nt * nf
    if random_seed:
        print('Using the following random_seed for dataset noise:',
              random_seed)
        np.random.seed(random_seed)
    real_noise = np.random.normal(0, sigma/2.**0.5, ndata)

    if random_seed:
        np.random.seed(random_seed*123)
    imag_noise = np.random.normal(0, sigma/2.**0.5, ndata)
    complex_noise = real_noise + 1j*imag_noise
    complex_noise = complex_noise * sigma/complex_noise.std()
    complex_noise_hermitian = complex_noise.copy()

    """
    How to create a conjugate baseline map from the instrument model:

    1. Create a map for a single time step that maps the array indices
       of baselines with (u, v) and (-u, -v)
    2. Add noise to (u, v) and conjugate noise to (-u, -v) using the
       map from step 1 per time and frequency (identical map can be used
       at all frequencies).
    """
    bl_conjugate_pairs_dict = {}
    bl_conjugate_pairs_map = {}
    # Only account for uv-redundancy for now so use
    # uvw_array_meters[:,:2] and exclude w-coordinate
    for i, uvw in enumerate(uvw_array_meters[:, :2]):
        if tuple(uvw*-1) in bl_conjugate_pairs_dict.keys():
            key = bl_conjugate_pairs_dict[tuple(uvw*-1)]
            bl_conjugate_pairs_dict[tuple(uvw)] = key
            bl_conjugate_pairs_map[key] = i
        else:
            bl_conjugate_pairs_dict[tuple(uvw)] = i

    for i_t in range(nt):
        time_ind = i_t * nbls * nf
        for i_freq in range(nf):
            freq_ind = i_freq * nbls
            start_ind = time_ind + freq_ind
            for bl_ind in bl_conjugate_pairs_map.keys():
                conj_bl_ind = bl_conjugate_pairs_map[bl_ind]
                complex_noise_hermitian[start_ind+conj_bl_ind] =\
                    complex_noise_hermitian[start_ind+bl_ind].conjugate()
            complex_noise_hermitian[start_ind : start_ind+nbls] /=\
                bl_redundancy_array[:, 0]**0.5

    d = s + complex_noise_hermitian.flatten()

    return d, complex_noise_hermitian.flatten(), bl_conjugate_pairs_map


def generate_masked_coordinate_cubes(
        cube_to_mask, nu, nv, nf, neta, nq,
        ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc):
    # Generate k_cube physical coordinates
    # to match the 21cmFAST input simulation
    mod_k, k_x, k_y, k_z, x, y, z =\
        generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
            nu, nv, nf, neta,
            ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc)

    # Do not include high spatial frequency structure in the power
    # spectral data since these terms aren't included in the data model
    Nyquist_k_z_mode = k_z[0, 0, 0]
    Second_highest_frequency_k_z_mode = k_z[-1, 0, 0]
    # NOTE: the k_z=0 term should not necessarily be masked out since it
    # is still required as a quadratic component (and is not currently
    # explicitly added in there) even if it is not used for calculating
    # the power spectrum.
    if nq == 1:
        high_spatial_frequency_selector_mask = k_z == Nyquist_k_z_mode
    else:
        high_spatial_frequency_selector_mask = np.logical_or.reduce(
            (k_z == Nyquist_k_z_mode,
             k_z == Second_highest_frequency_k_z_mode))
    high_spatial_frequency_mask = np.logical_not(
        high_spatial_frequency_selector_mask)

    if p.include_instrumental_effects:
        high_spatial_frequency_mask = (
                high_spatial_frequency_mask
                == high_spatial_frequency_mask)
        high_spatial_frequency_selector_mask = (
                high_spatial_frequency_selector_mask
                != high_spatial_frequency_selector_mask)

    Mean_k_z_mode = 0.0
    k_z_mean_mask = k_z != Mean_k_z_mode

    k_perp_3D = (k_x**2. + k_y**2)**0.5
    if p.fit_for_monopole:
        ZM_mask = k_perp_3D >= 0.0 # Don't exclude the mean from the fit
    else:
        ZM_mask = k_perp_3D > 0.0 # Exclude (u,v)=(0,0)
    ZM_selector_mask = np.logical_not(ZM_mask)

    ZM_2D_mask_vis_ordered = ZM_mask.T.flatten()
    high_spatial_frequency_mask_vis_ordered = np.logical_not(
        high_spatial_frequency_selector_mask.T.flatten())

    if nq > 0:
        ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and(
            high_spatial_frequency_mask_vis_ordered, ZM_2D_mask_vis_ordered)
    else:
        ZM_2D_and_high_spatial_frequencies_mask_vis_ordered =\
            ZM_2D_mask_vis_ordered

    model_cube_to_mask_vis_ordered = cube_to_mask.T.flatten()[
        ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]

    if nq > 0:
        model_cube_to_mask_vis_ordered_reshaped =\
            model_cube_to_mask_vis_ordered.reshape(-1, neta)

        WQ_boolean_array = np.zeros(
            [model_cube_to_mask_vis_ordered_reshaped.shape[0], nq]
            ).astype('bool')
        WQ_inf_array = (
                np.zeros(
                    [model_cube_to_mask_vis_ordered_reshaped.shape[0], nq])
                + np.inf)
        model_cube_to_mask_vis_ordered_reshaped_WQ = np.hstack(
            (model_cube_to_mask_vis_ordered_reshaped, WQ_inf_array))

        model_cube_to_mask_vis_ordered_WQ =\
            model_cube_to_mask_vis_ordered_reshaped_WQ.flatten()

        k_z_mean_mask_vis_ordered = np.logical_not(k_z_mean_mask).T[
            np.logical_and(
                ZM_mask.T,
                np.logical_not(high_spatial_frequency_selector_mask).T)]
        k_z_mean_mask_vis_ordered_reshaped =\
            k_z_mean_mask_vis_ordered.reshape(
                model_cube_to_mask_vis_ordered_reshaped.shape)

        Quad_modes_only_boolean_array_vis_ordered = np.hstack(
            (k_z_mean_mask_vis_ordered_reshaped.astype('bool'),
             np.logical_not(WQ_boolean_array))).flatten()

        return model_cube_to_mask_vis_ordered_WQ
    else:
        return model_cube_to_mask_vis_ordered


def generate_k_cube_model_spherical_binning_v2d1(
        mod_k_masked, k_z_masked, nu, nv, nf, neta, nq,
        ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc):
    # Need to rename this function, it's now the default
    # Generate k_cube physical coordinates
    # to match the 21cmFAST input simulation
    mod_k, k_x, k_y, k_z, x, y, z =\
        generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
            nu, nv, nf, neta, ps_box_size_ra_Mpc,
            ps_box_size_dec_Mpc, ps_box_size_para_Mpc)

    modkscaleterm = 1.35
    deltakpara = 2.*np.pi / ps_box_size_para_Mpc
    binsize = deltakpara * 2.0

    numKbins = 50
    modkbins = np.zeros([numKbins, 2])
    modkbins[0, 0] = 0
    modkbins[0, 1] = binsize

    for m1 in range(1, numKbins, 1):
        binsize = binsize * modkscaleterm
        modkbins[m1, 0] = modkbins[m1-1, 1]
        modkbins[m1, 1] = modkscaleterm * modkbins[m1, 0]

    total_elements = 0
    n_bins = 0
    modkbins_containing_voxels = []
    for i_bin in range(numKbins):
        #NOTE: By requiring k_z>0 the constant term in the 1D FFT is now
        # effectively a quadratic mode! If it is to be included
        # explicitly with the quadratic modes, then k_z==0 should be
        # added to the quadratic selector mask
        n_elements = np.sum(
            np.logical_and.reduce(
                (mod_k_masked > modkbins[i_bin, 0],
                 mod_k_masked <= modkbins[i_bin, 1],
                 k_z_masked != 0)
                )
            )
        if n_elements > 0:
            n_bins += 1
            # print(i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements)
            total_elements += n_elements
            modkbins_containing_voxels.append((modkbins[i_bin], n_elements))
    # print(total_elements, mod_k_masked.size)

    k_cube_voxels_in_bin = []
    count = 0
    for i_bin in range(len(modkbins_containing_voxels)):
        relevant_voxels = np.where(
            np.logical_and.reduce(
                (mod_k_masked > modkbins_containing_voxels[i_bin][0][0],
                 mod_k_masked <= modkbins_containing_voxels[i_bin][0][1],
                 k_z_masked != 0)
                )
            )
        # print(relevant_voxels)
        # print(len(relevant_voxels[0]))
        count += len(relevant_voxels[0])
        k_cube_voxels_in_bin.append(relevant_voxels)
    # print(count) #should be mod_k_masked.shape[0]-3*nuv

    return k_cube_voxels_in_bin, modkbins_containing_voxels


def calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, **kwargs):
    # ===== Defaults =====
    default_save_k_vals = False
    default_k_vals_file = 'k_vals.txt'
    default_k_vals_dir = 'k_vals'

    # ===== Inputs =====
    save_k_vals = kwargs.pop('save_k_vals',default_save_k_vals)
    k_vals_file = kwargs.pop('k_vals_file',default_k_vals_file)
    k_vals_dir = kwargs.pop('k_vals_dir',default_k_vals_dir)

    k_vals = []
    kbin_edges = []
    nsamples = []
    print('\n---Calculating k-vals---')
    for i_bin in range(len(k_cube_voxels_in_bin)):
        mean_mod_k = mod_k_masked[k_cube_voxels_in_bin[i_bin]].mean()
        min_k = mod_k_masked[k_cube_voxels_in_bin[i_bin]].min()
        kbin_edges.append(min_k)
        nsamples.append(len(k_cube_voxels_in_bin[i_bin][0]))
        k_vals.append(mean_mod_k)
        print(i_bin, mean_mod_k)
    max_k = mod_k_masked[k_cube_voxels_in_bin[i_bin]].max()
    kbin_edges.append(max_k)

    if save_k_vals:
        if not os.path.exists(k_vals_dir):
            print('Directory not found: \n\n' + k_vals_dir + "\n")
            print('Creating required directory structure..')
            os.makedirs(k_vals_dir)

        np.savetxt(k_vals_dir + '/' + k_vals_file, k_vals)
        np.savetxt(k_vals_dir + '/'
                   + k_vals_file.replace('.txt', '_bins.txt'),
                   kbin_edges)
        np.savetxt(k_vals_dir + '/'
                   + k_vals_file.replace('.txt', '_nsamples.txt'),
                   nsamples)

    return np.array(k_vals)


def generate_k_cube_model_cylindrical_binning(
        mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins,
        nu, nv, nf, neta, nq, ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc, ps_box_size_para_Mpc):
    # Generate k_cube physical coordinates to
    # match the 21cmFAST input simulation
    mod_k, k_x, k_y, k_z, x, y, z =\
        generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
            nu, nv, nf, neta,
            ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc)

    # define mod_k binning
    modkscaleterm = 1.5 # Value used in BEoRfgs and in 21cmFAST binning
    deltakperp = 2.*np.pi / ps_box_size_para_Mpc
    binsize = deltakperp * 2 # Value used in BEoRfgs

    numKbins = 50
    modkbins = np.zeros([numKbins, 2])
    modkbins[0, 0] = 0
    modkbins[0, 1] = binsize

    for m1 in range(1, numKbins, 1):
        binsize = binsize * modkscaleterm
        modkbins[m1, 0] = modkbins[m1-1, 1]
        modkbins[m1, 1] = modkscaleterm * modkbins[m1, 0]

    total_elements = 0
    n_bins = 0
    modkbins_containing_voxels = []
    for i_bin in range(numKbins):
        # NOTE: By requiring k_z>0 the constant term in the 1D FFT is
        # now effectively a quadratic mode! If it is to be included
        # explicitly with the quadratic modes,, then k_z==0 should be
        # added to the quadratic selector mask
        n_elements = np.sum(
            np.logical_and.reduce(
                (mod_k_masked > modkbins[i_bin, 0],
                 mod_k_masked <= modkbins[i_bin, 1],
                 k_z_masked > 0)
                )
            )
        if n_elements > 0:
            n_bins += 1
            # print(i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements)
            total_elements += n_elements
            modkbins_containing_voxels.append((modkbins[i_bin], n_elements))
    # print(total_elements, mod_k_masked.size)

    # define k_perp binning
    k_perp_3D = (k_x_masked**2. + k_y_masked**2)**0.5
    k_perp_min = k_perp_3D[np.isfinite(k_perp_3D)].min()
    k_perp_max = k_perp_3D[np.isfinite(k_perp_3D)].max()

    def output_k_perp_bins(k_perp_min, k_perp_max, n_k_perp_bins):
        k_perp_bins = np.vstack(
            (np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins)[:-1],
             np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins)[1:])
            ).T
        return k_perp_bins

    def return_k_perp_bins_with_voxels(k_perp_bins):
        total_elements = 0
        n_bins = 0
        k_perp_bins_containing_voxels = []
        for i_bin in range(len(k_perp_bins)):
            # NOTE: By requiring k_z>0 the constant term in the 1D FFT
            # is now effectively a quadratic mode! If it is to be
            # included explicitly with the quadratic modes, then k_z==0
            # should be added to the quadratic selector mask
            k_perp_constraint = np.logical_and.reduce(
                (k_perp_3D > k_perp_bins[i_bin][0],
                 k_perp_3D <= k_perp_bins[i_bin][1]))
            n_elements = np.sum(k_perp_constraint)
            if n_elements > 0:
                n_bins += 1
                # print(i_bin,
                #       k_perp_bins[i_bin,0],
                #       k_perp_bins[i_bin,1],
                #       n_elements)
                total_elements += n_elements
                k_perp_bins_containing_voxels.append(k_perp_bins[i_bin])
        # Note: total_elements should be mod_k_masked.size-2*nuv since
        # it doesn't include the linear and quadratic mode channels
        # print(total_elements, mod_k_masked.size)

        return k_perp_bins_containing_voxels

    n_k_perp_bins_array = [n_k_perp_bins
                           for _ in range(len(modkbins_containing_voxels))]

    # Currently using equal bin widths for all k_z but can make the
    # number of bins / the bin width a function of k_z if that turns out
    # to be useful (i.e. use larger bins at larger k_z where there is
    # less power).
    k_perp_bins = [return_k_perp_bins_with_voxels(
        output_k_perp_bins(k_perp_min, k_perp_max, n_k_perp_bins_val))
        for n_k_perp_bins_val in n_k_perp_bins_array]
    # k_perp_bins = [np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins)
    #                for _ in range(len(modkbins_containing_voxels))]

    k_cube_voxels_in_bin = []
    count = 0
    for i_mod_k_bin in range(len(modkbins_containing_voxels)):
        for j_k_perp_bin in range(len(k_perp_bins[0])):
            # Since calculating the cylindrical power spectrum - bin in
            # k_z rather than mod_k. However maintain the mod_k bin
            # limits for the k_z binning for approximate bin size
            # consistency with previous results when calculating the
            # cylindrical power spectrum with HERA 37.
            k_z_constraint = np.logical_and.reduce(
                (abs(k_z_masked)
                 > modkbins_containing_voxels[i_mod_k_bin][0][0],
                 abs(k_z_masked)
                 <= modkbins_containing_voxels[i_mod_k_bin][0][1],
                 k_z_masked != 0)
                )
            k_perp_constraint = np.logical_and.reduce(
                (k_perp_3D > k_perp_bins[i_mod_k_bin][j_k_perp_bin][0],
                 k_perp_3D <= k_perp_bins[i_mod_k_bin][j_k_perp_bin][1]))
            relevant_voxels = np.where(
                np.logical_and(k_z_constraint, k_perp_constraint))
            # print(relevant_voxels)
            # print(len(relevant_voxels[0]))
            count += len(relevant_voxels[0])
            k_cube_voxels_in_bin.append(relevant_voxels)
    print(count) # should be mod_k_masked.shape[0]-3*nuv

    return k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins
