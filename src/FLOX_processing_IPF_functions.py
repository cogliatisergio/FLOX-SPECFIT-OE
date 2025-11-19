import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.distance import cosine
from scipy.special import gamma, gammainc
import scipy
import xarray as xr
import logging
import warnings


def _get_red_sif(wl, sif):
    """
    Extract red fluorescence peak at 684 nm.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_R_max, SIF_R_wl) - Red peak intensity and wavelength
    """
    index = np.argmin(np.abs(wl - 684))  # TODO add consts
    sif_r_max = sif[index]
    if np.isnan(sif_r_max):
        sif_r_wl = np.nan
    else:
        sif_r_wl = wl[
            14
        ]  # TODO MAGIC shouldn't be here but local max too hard to find because second peak is hiding it.
    return sif_r_max, sif_r_wl


def _get_far_red_sif(wl, sif):
    """
    Extract far-red fluorescence peak (>720 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_FR_max, SIF_FR_wl) - Far-red peak intensity and wavelength
    """
    mask_far_red = wl > 720  # TODO add const
    max_far_red_index = np.argmax(sif[mask_far_red])
    sif_fr_max = sif[mask_far_red][max_far_red_index]
    if np.isnan(sif_fr_max):
        sif_fr_wl = np.nan
    else:
        sif_fr_wl = wl[mask_far_red][max_far_red_index]
    return sif_fr_max, sif_fr_wl


def _get_o2a_sif(wl, sif):
    """
    Extract SIF value at O2-A absorption line (760 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 760 nm
    """
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii]
    return sif_o2a


def _get_o2b_sif(wl, sif):
    """
    Extract SIF value at O2-B absorption line (687 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 687 nm
    """
    index = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[index]
    return sif_o2b


def _get_spectrally_integrated_sif(wl, sif):
    """
    Calculate total SIF by spectral integration.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        Integrated SIF value [W m⁻² sr⁻¹]
    """
    sifint = np.trapezoid(sif, wl)
    return sifint


def _reflectance_concatenation(
    floris_app_refl_map, floris_wv_merged, rhomin_wl, wvl_1nm, min_wl
):
    """
    Concatenate interpolated reflectance with minimum wavelength data.

    Parameters
    ----------
    floris_app_refl_map : numpy.ndarray
        FLORIS apparent reflectance values
    FLORIS_wv_merged : numpy.ndarray
        FLORIS wavelength grid [nm]
    RHOmin_wl : numpy.ndarray
        Reflectance data for minimum wavelength range
    wvl_1nm : numpy.ndarray
        Target 1nm wavelength grid [nm]
    min_wl : float
        Minimum wavelength threshold [nm]

    Returns
    -------
    numpy.ndarray
        Concatenated reflectance spectrum
    """
    # filter out zero and nan values from floris_wv_merged
    idx = (floris_wv_merged != 0) & (~np.isnan(floris_wv_merged))

    # interpolate floris_app_refl_map at points wvl_1nm using filtered floris_wv_merged
    tmp_rho = np.interp(wvl_1nm, floris_wv_merged[idx], floris_app_refl_map[idx])

    # concatenate values where wvl_1nm < min_wl with rhomin_wl
    y = np.concatenate((tmp_rho[wvl_1nm < min_wl], rhomin_wl))
    return y


def _compute_apparent_reflectance(x, sp, wvl):
    """
    Compute apparent reflectance using spline interpolation.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
        Shape (n,) for scalar or (m,n) for vector mode
    sp : Spline object
        Spline interpolator with settable coefficients
    wvl : numpy.ndarray
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray
        Reflectance values at input wavelengths

    Notes
    -----
    Only parameters x[8:] affect reflectance. In vector mode,
    first 8 rows are copied from row 7 for efficiency.
    """
    if len(x.shape) == 1:  # Scalar mode
        sp.c = x[8:].flatten()
        rho = sp(wvl)
    else:  # Vector mode (is only used for the jacobian calculation)
        rho = np.zeros((x.shape[0], wvl.shape[0]), dtype=np.float64)
        for i in range(7, x.shape[1]):
            sp.c = x[i, 8:].flatten()
            rho[i, :] = sp(wvl)
        rho[:7, :] = np.tile(rho[7, :], (7, 1))
        rho[-1, :] = rho[7, :]
    return rho


def _compute_fluorescence(x, sp, wvl):
    """
    Compute Solar-Induced Fluorescence using forward model.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
        Shape (n,) for scalar or (m,n) for vector mode
    sp : Spline object
        Unused, kept for interface consistency
    wvl : numpy.ndarray
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray
        Fluorescence spectrum at input wavelengths

    Notes
    -----
    Only parameters x[:8] affect fluorescence. In vector mode,
    rows 8+ are copied from row 8 for efficiency.
    """
    if len(x.shape) == 1:
        fluorescence = _sif_forward_model(x[:8], wvl)
    else:
        fluorescence = np.zeros((x.shape[0], wvl.shape[0]), dtype=np.float64)
        for i in range(9):
            sp.c = x[i, :8]
            fluorescence[i, :] = _sif_forward_model(x[i, :8], wvl)
        fluorescence[8:, :] = np.tile(fluorescence[8, :], (19, 1))

    return fluorescence


def l2b_forward_model(x, wvl, sp, te, tes, lp, tup, ts):
    """
    Level 2B forward model for atmospheric radiative transfer with fluorescence.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
    wvl : numpy.ndarray
        Wavelength vector [nm]
    sp : Spline object
        Surface reflectance interpolator
    te, tes, lp, tup, ts : numpy.ndarray
        Atmospheric transmission and path parameters

    Returns
    -------
    numpy.ndarray
        Apparent reflectance (ARHO_SIM) as observed by sensor

    Notes
    -----
    Implements: L_SIM = LP + (TE/π)·ρ + TUP·F + (TES/π)·ρ² + TS·F·ρ
    Then inverts to get apparent reflectance using quadratic formula.
    """
    rho = _compute_apparent_reflectance(x, sp, wvl)

    fluorescence = _compute_fluorescence(x, sp, wvl)

    # Calculate L_SIM using element-wise operations
    l_sim = (
        lp
        + (te / np.pi) * rho
        + tup * fluorescence
        + (tes / np.pi) * rho**2
        + ts * fluorescence * rho
    )

    # Calculate ARHO_SIM using quadratic formula
    discriminant = te**2 - 4 * tes * np.pi * (lp - l_sim)
    arho_sim = (-te + np.sqrt(discriminant)) / (2 * tes)

    return arho_sim


def _sif_forward_model(parameters, wavelengths):
    """
    Solar-Induced Fluorescence forward model using dual-peak approach.

    This function models SIF spectra using two distinct fluorescence peaks:
    1. Red fluorescence peak: Modeled using a Lorentzian-like function
    2. Far-red fluorescence peak: Modeled using an asymmetric super-Gaussian function

    Parameters
    ----------
    parameters : numpy.ndarray, shape (8, n) or (8,)
        Model parameters where each row represents:
        [0] : Red peak amplitude (I_red)
        [1] : Red peak center wavelength (λ_red) [nm]
        [2] : Red peak width parameter (σ_red) [nm]
        [3] : Far-red peak intensity (I_far)
        [4] : Far-red peak center wavelength (C_far) [nm]
        [5] : Far-red peak base width (w_far) [nm]
        [6] : Far-red peak shape parameter (k_far) [-]
        [7] : Far-red peak asymmetry width (aw_far) [nm]

    wavelengths : numpy.ndarray, shape (m,)
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray, shape (m, n)
        Modeled fluorescence spectrum where m is the number of wavelengths
        and n is the number of parameter sets
    """
    # Ensure input is 2D for vectorized operations
    if parameters.ndim == 1:
        parameters = parameters.reshape(-1, 1)

    # Extract and validate parameters
    red_amplitude = parameters[0]  # I_red
    red_center = parameters[1]  # λ_red [nm]
    red_width = parameters[2]  # σ_red [nm]

    far_red_intensity = parameters[3]  # I_far
    far_red_center = parameters[4]  # C_far [nm]
    far_red_base_width = parameters[5]  # w_far [nm]
    far_red_shape = np.abs(parameters[6])  # k_far (ensure positive)
    far_red_asymmetry = parameters[7]  # aw_far [nm]

    wavelength_deviation = (wavelengths - red_center) / red_width
    red_fluorescence = red_amplitude / (wavelength_deviation**2 + 1)

    # Model Far-Red Fluorescence Peak using Asymmetric Super-Gaussian
    far_red_fluorescence = _compute_asymmetric_super_gaussian(
        wavelengths=wavelengths,
        intensity=far_red_intensity,
        center=far_red_center,
        base_width=far_red_base_width,
        shape_parameter=far_red_shape,
        asymmetry_width=far_red_asymmetry,
    )

    # Combine fluorescence components
    total_fluorescence = red_fluorescence + far_red_fluorescence

    return total_fluorescence


def _compute_asymmetric_super_gaussian(
    wavelengths, intensity, center, base_width, shape_parameter, asymmetry_width
):
    """
    Compute asymmetric super-Gaussian fluorescence peak.

    This function implements the asymmetric super-Gaussian formulation where
    different widths are applied on the left and right sides of the peak center.
    The barycenter is analytically calculated to ensure proper peak positioning.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Wavelength vector [nm]
    intensity : float or numpy.ndarray
        Peak intensity
    center : float or numpy.ndarray
        Peak center wavelength [nm]
    base_width : float or numpy.ndarray
        Base width parameter [nm]
    shape_parameter : float or numpy.ndarray
        Shape parameter k (controls peak flatness)
    asymmetry_width : float or numpy.ndarray
        Asymmetry width parameter [nm]

    Returns
    -------
    numpy.ndarray
        Asymmetric super-Gaussian values at input wavelengths
    """
    warnings.filterwarnings("error")
    # Apply asymmetry constraint
    asymmetry_width = np.where(base_width < asymmetry_width, 0, asymmetry_width)

    # Calculate barycenter offset
    barycenter_offset = _analytical_barycenter(
        center,
        base_width,
        shape_parameter,
        asymmetry_width,
        wavelengths.min(),
        wavelengths.max(),
    )

    # Precompute inverse widths
    left_width = base_width - asymmetry_width
    right_width = base_width + asymmetry_width
    inv_left = 1.0 / left_width
    inv_right = 1.0 / right_width

    # Vectorized domain calculation
    adjusted_center = center - barycenter_offset
    wavelength_offset = wavelengths - adjusted_center
    width_inv = np.where(wavelength_offset <= 0, inv_left, inv_right)

    # Single exponential evaluation
    scaled_offset = wavelength_offset * width_inv
    exponent = -(np.abs(scaled_offset) ** shape_parameter)
    component = np.exp(exponent)
    warnings.resetwarnings()
    return intensity * component


def _analytical_barycenter(c, w, k, aw, wvl_min, wvl_max):
    """
    Vectorized implementation of analytical barycenter calculation
    """
    # Prevent division by zero in gamma functions
    k = np.clip(k, 1e-10, None)

    arg1 = ((c - wvl_min) / (w - aw)) ** k
    arg2 = ((wvl_max - c) / (w + aw)) ** k
    g1 = gamma(2 / k) * gammainc(2 / k, arg1)
    g2 = gamma(2 / k) * gammainc(2 / k, arg2)
    g3 = gamma(1 / k) * gammainc(1 / k, arg1)
    g4 = gamma(1 / k) * gammainc(1 / k, arg2)

    # Calculate m with numerical stability
    numerator = (w + aw) ** 2 * g2 - (w - aw) ** 2 * g1
    denominator = (w + aw) * g4 + (w - aw) * g3
    m = numerator / denominator  # np.clip(denominator, 1e-10, None)

    return m


def _l2b_regularized_cost_function_optimization(
    wvl,
    apparent_reflectance,
    xa_mean,
    atm_func,
    sa,
    sy,
    g,
    l2b_wavelength_grid,
    max_iter=15,
    ftol=1e-4,
):
    knots = np.array(
        [
            wvl[0],
            wvl[0],
            wvl[0],
            wvl[0],
            675.0000,
            682.6000,
            693.4500,
            695.4500,
            699.0333,
            704.4500,
            708.5,
            712.1000,
            734.6333,
            738.5,
            743.5,
            747.8333,
            755.5000,
            771.000,
            wvl[-1],
            wvl[-1],
            wvl[-1],
            wvl[-1],
        ]
    )
    sp = BSpline(knots, xa_mean[8:], 3)

    sp.c = xa_mean[8:]  # update weights

    def cost_function(fx, x):
        c = (apparent_reflectance - fx).T @ sy @ (apparent_reflectance - fx) + g * (
            x - xa_mean
        ).T @ sa @ (x - xa_mean)
        return c

    def fw(x):
        return l2b_forward_model(
            x,
            wvl,
            sp,
            te=atm_func["TE"],
            tes=atm_func["TES"],
            lp=atm_func["Lp0"],
            tup=atm_func["T"],
            ts=atm_func["TS"],
        )

    def jac(x):
        return _jacobian_calculation(fw, x)

    lm_gamma = 0.1

    y = apparent_reflectance
    x0 = xa_mean.copy()
    x1 = xa_mean.copy()

    fx0 = fw(x0)
    k = jac(x0)
    c0 = (y - fx0) @ sy @ (y - fx0) + g * (x0 - xa_mean) @ sa @ (x0 - xa_mean)
    i = 0
    c = ftol
    while c >= ftol and i < max_iter:
        i += 1
        ci = g * sa + k.T @ sy @ k + lm_gamma * sa
        x1 = x0 + scipy.linalg.inv(ci) @ (
            k.T @ (sy @ (y - fx0)) - g * sa @ (x0 - xa_mean)
        )
        fx1 = fw(x1)
        k1 = jac(x1)
        c1 = cost_function(fx1, x1)
        if c1 >= c0:
            x1 = x0
            fx1 = fx0
            c1 = c0
            k1 = k
            lm_gamma = lm_gamma / 10
            continue
        else:
            t0 = 1 - (np.linalg.norm(y - fx1) / np.linalg.norm(y - fx0)) ** 2
            t1 = (np.linalg.norm(k * (x1 - x0)) / np.linalg.norm(y - fx0)) ** 2
            t2 = (
                2
                * (
                    (np.sqrt(lm_gamma) * np.linalg.norm(np.eye(26) * (x1 - x0)))
                    / np.linalg.norm(y - fx0)
                )
                ** 2
            )
            rho = t0 / (t1 + t2)

            if rho > 0.01:
                lm_gamma = lm_gamma * 1.1
            else:
                lm_gamma = lm_gamma / 10
        c = abs(c1 - c0) / c0 * 1e2
        x0 = x1
        fx0 = fx1
        c0 = c1
        k = k1

    sp.c = x1[8:]
    reflectance = sp(l2b_wavelength_grid)
    sif = _sif_forward_model(x1[:8].flatten(), l2b_wavelength_grid)
    return reflectance, sif


def _jacobian_calculation(func, x, epsilon=np.float64(1e-6)):
    num_parameters = len(x)
    x_perturbed = np.tile(x, (num_parameters + 1, 1)) + np.concatenate(
        [epsilon * np.eye(num_parameters), np.zeros((1, num_parameters))]
    )
    y = func(x_perturbed)
    return ((y[:-1, :] - y[-1, :]) * (1 / epsilon)).T