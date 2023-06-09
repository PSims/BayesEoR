import numpy as np

from ..utils import mpiprint


def generate_data_and_noise_vector_instrumental(
        sigma, s, nf, nt, uvw_array_meters, bl_redundancy_array,
        random_seed='', rank=0):
    """
    Creates a noise vector (n), with Hermitian structure based upon the
    uv sampling in the instrument model, and adds this noise to the input,
    noiseless visibilities (s) to form the data vector d = s + n.

    Parameters
    ----------
    sigma : float
        Noise amplitude of |n|^2.  The complex amplitude is calculated as
        sigma/sqrt(2).
    s : np.ndarray of complex floats
        Input signal (visibilities).
    nf : int
        Number of frequency channels.
    nt : int
        Number of times.
    uvw_array_meters : np.ndarray of floats
        Instrument model uv-sampling with shape (nbls, 3).
    bl_redundancy_array : np.ndarray of floats
        Number of baselines per redundant baseline group.
    random_seed : int
        Used to seed `np.random` when generating the noise vector.
    rank : int
        MPI rank.

    Returns
    -------
    d : np.ndarray of complex floats
        Data vector of complex signal + noise visibilities.
    complex_noise_hermitian : np.ndarray of complex floats
        Vector of complex noise amplitudes.
    bl_conjugate_pairs_map : dictionary
        Dictionary containing the array index mapping of conjugate baseline
        pairs based on `uvw_array_meters`.

    """
    if sigma == 0.0:
        complex_noise_hermitian = np.zeros(len(s)) + 0.0j
    else:
        nbls = len(uvw_array_meters)
        ndata = nbls * nt * nf
        if random_seed:
            mpiprint(f"Seeding numpy.random with {random_seed}", rank=rank)
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
            complex_noise_hermitian[start_ind:start_ind+nbls] /=\
                bl_redundancy_array[:, 0]**0.5

        d = s + complex_noise_hermitian.flatten()

    return d, complex_noise_hermitian.flatten(), bl_conjugate_pairs_map