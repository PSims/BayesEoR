import numpy as np
from pathlib import Path

from ..utils import mpiprint


def generate_k_cube_in_physical_coordinates(
        nu, nv, neta, ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc, ps_box_size_para_Mpc):
    """
    Generates rectilinear k-space cubes in units of inverse Mpc.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    ps_box_size_ra_Mpc : float
        Right Ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.

    Returns
    -------
    mod_k_physical : np.ndarray of floats
        Modulus of each 3D k-space voxel, i.e. sqrt(k_x**2 + k_y**2 + kz**2).
    k_x : np.ndarray of floats
        Array of RA axis Fourier modes in inverse Mpc.
    k_y : np.ndarray of floats
        Array of DEC axis Fourier modes in inverse Mpc.
    k_z : np.ndarray of floats
        Array of LoS axis Fourier modes in inverse Mpc.
    x : np.ndarray of ints
        Array of RA axis Fourier mode pixel coordinates.
    y : np.ndarray of ints
        Array of DEC axis Fourier mode pixel coordinates.
    z : np.ndarray of ints
        Array of LoS axis Fourier mode pixel coordinates.

    """
    # Generate k_cube pixel coordinates
    eta_ind_min = -(neta//2)
    eta_ind_max = neta//2
    if neta % 2 == 1:
        # neta is odd
        eta_ind_max += 1
    z, y, x = np.mgrid[eta_ind_min:eta_ind_max,
                       -(nv//2):(nv//2)+1,
                       -(nu//2):(nu//2)+1]

    # Setup k-space arrays
    deltakx = 2.*np.pi / ps_box_size_ra_Mpc
    deltaky = 2.*np.pi / ps_box_size_dec_Mpc
    deltakz = 2.*np.pi / ps_box_size_para_Mpc
    k_z = z * deltakz
    k_y = y * deltaky
    k_x = x * deltakx
    mod_k_physical = (k_z**2. + k_y**2. + k_x**2.)**0.5

    return mod_k_physical, k_x, k_y, k_z, x, y, z


def mask_k_cube(mod_k):
    """
    Creates a mask and masks Fourier and Large Spectral Scale Model (LSSM)
    modes that are unused for estimating the power spectrum.

    Parameters
    ----------
    mod_k : np.ndarray of floats
        Modulus of each 3D k-space voxel, i.e. sqrt(k_x**2 + k_y**2 + kz**2).

    Returns
    -------
    mod_k_vo : np.ndarray
        Vis-ordered `mod_k`.

    """
    neta, nv, nu = mod_k.shape

    # Remove the eta=0 mode from the EoR k-cube
    mod_k = np.delete(mod_k, neta//2, axis=0)

    # Flatten along k_x then k_y
    mod_k = mod_k.reshape((neta - 1, nu*nv))

    # Remove (u, v) = (0, 0)
    mod_k = np.delete(mod_k, (nu*nv)//2, axis=1)

    # Flattened arrays move first along k_z, then k_x, and lastly k_y
    mod_k_vo = mod_k.flatten(order='F')

    return mod_k_vo


def generate_k_cube_model_spherical_binning(mod_k_vo, ps_box_size_para_Mpc):
    """
    Generates a set of spherical k-space bins from which the 1D power spectrum
    is calculated.

    Parameters
    ----------
    mod_k_vo : np.ndarray
        Vis-ordered array of |k| = sqrt(k_x**2 + k_y**2 + k_z**2).  Should only
        contain the k values used in EoR power spectrum estimation.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.

    Returns
    -------
    k_cube_voxels_in_bin : list
        List containing sublists for each spherically averaged k-bin.  Each
        sublist contains the flattened 3D k-space cube index of all |k| that
        fall within a given spherical k-bin.
    modkbins_containing_voxels : list
        Number of |k| that fall within each spherical k-bin.

    """
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
        # NOTE: By requiring k_z>0 the constant term in the 1D FFT is now
        # effectively a quadratic mode! If it is to be included
        # explicitly with the quadratic modes, then k_z==0 should be
        # added to the quadratic selector mask
        n_elements = np.sum(
            np.logical_and.reduce((
                mod_k_vo > modkbins[i_bin, 0],
                mod_k_vo <= modkbins[i_bin, 1]
            ))
        )
        if n_elements > 0:
            n_bins += 1
            total_elements += n_elements
            modkbins_containing_voxels.append((modkbins[i_bin], n_elements))

    k_cube_voxels_in_bin = []
    count = 0
    for i_bin in range(len(modkbins_containing_voxels)):
        relevant_voxels = np.where(
            np.logical_and.reduce((
                mod_k_vo > modkbins_containing_voxels[i_bin][0][0],
                mod_k_vo <= modkbins_containing_voxels[i_bin][0][1]
            ))
        )
        count += len(relevant_voxels[0])
        k_cube_voxels_in_bin.append(relevant_voxels)

    return k_cube_voxels_in_bin, modkbins_containing_voxels


def calc_mean_binned_k_vals(
    mod_k_vo, k_cube_voxels_in_bin, save_k_vals=False, clobber=False,
    k_vals_file="k-vals.txt", k_vals_dir="k_vals", rank=0
):
    """
    Calculates the mean of all |k| that fall within a k-bin.

    Parameters
    ----------
    mod_k_vo : np.ndarray
        Vis-ordered array of |k| = sqrt(k_x**2 + k_y**2 + k_z**2).
    k_cube_voxels_in_bin : list
        List containing sublists for each k-bin.  Each sublist contains the
        flattened 3D k-space cube index of all |k| that fall within a given
        k-bin.
    save_k_vals : bool
        If `True`, save mean k values to `k_vals_file`.
    clobber : bool
        If `True`, overwrite existing files.
    k_vals_file : str
        Filename for saved k values.  Defaults to 'k-vals.txt'.
    k_vals_dir : str
        Directory in which to save k values.  Defaults to './k_vals/'.
    rank : int
        MPI rank.

    Returns
    -------
    k_vals : np.ndarray of floats
        Array containing the mean k for each k-bin.

    """
    k_vals = []
    kbin_edges = []
    nsamples = []
    mpiprint("\n---Calculating k-vals---", rank=rank)
    for i_bin in range(len(k_cube_voxels_in_bin)):
        mean_mod_k = mod_k_vo[k_cube_voxels_in_bin[i_bin]].mean()
        min_k = mod_k_vo[k_cube_voxels_in_bin[i_bin]].min()
        kbin_edges.append(min_k)
        nsamples.append(len(k_cube_voxels_in_bin[i_bin][0]))
        k_vals.append(mean_mod_k)
        mpiprint(i_bin, mean_mod_k, rank=rank)
    max_k = mod_k_vo[k_cube_voxels_in_bin[i_bin]].max()
    kbin_edges.append(max_k)
    mpiprint("", rank=rank)

    k_vals_dir = Path(k_vals_dir)
    save = save_k_vals * (not (k_vals_dir / k_vals_file).exists() or clobber)
    if save:
        if not k_vals_dir.exists():
            mpiprint(f"Directory not found: \n\n{k_vals_dir}\n", rank=rank)
            mpiprint("Creating required directory structure...", rank=rank)
            k_vals_dir.mkdir()

        np.savetxt(k_vals_dir / k_vals_file, k_vals)
        np.savetxt(
            k_vals_dir / k_vals_file.replace(".txt", "-bins.txt"), kbin_edges
        )
        np.savetxt(
            k_vals_dir / k_vals_file.replace(".txt", "-nsamples.txt"), nsamples
        )

    return np.array(k_vals)


def generate_k_cube_model_cylindrical_binning(
        mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins,
        ps_box_size_para_Mpc):
    """
    Generates a set of cylindrical k-space bins from which the 2D power
    spectrum is calculated.

    Parameters
    ----------
    mod_k_masked : np.ndarray of floats
        Array of |k| = sqrt(k_x**2 + k_y**2 + k_z**2) containing only modes
        used to estimate the power spectrum.
    k_z_masked : np.ndarray of floats
        Array of Line of Sight (LoS, frequency axis) Fourier modes used to
        estimate the power spectrum.
    k_y_masked : np.ndarray of floats
        Array of DEC axis Fourier modes used to estimate the power spectrum.
    k_x_masked : np.ndarray of floats
        Array of RA axis Fourier modes used to estimate the power spectrum.
    n_k_perp_bins : int
        Number of bins to make along the k-perpendicular axis where
        k_perp = sqrt(k_x**2 + k_y**2).
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.

    Returns
    -------
    k_cube_voxels_in_bin : list
        List containing sublists for each spherically averaged k-bin.  Each
        sublist contains the flattened 3D k-space cube index of all |k| that
        fall within a given spherical k-bin.
    modkbins_containing_voxels : list
        Number of |k| that fall within each spherical k-bin.
    k_perp_bins : list
        List of k_perp Fourier modes.

    """
    # define mod_k binning
    modkscaleterm = 1.5  # Value used in BEoRfgs and in 21cmFAST binning
    deltakperp = 2.*np.pi / ps_box_size_para_Mpc
    binsize = deltakperp * 2  # Value used in BEoRfgs

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
            total_elements += n_elements
            modkbins_containing_voxels.append((modkbins[i_bin], n_elements))

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
            count += len(relevant_voxels[0])
            k_cube_voxels_in_bin.append(relevant_voxels)

    return k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins
