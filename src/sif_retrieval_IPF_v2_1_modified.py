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


def _compute_reflectance(x, sp, wvl):
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


def l2b_forward_model(x, wvl, sp, Lin):
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

        
    Returns
    -------
    numpy.ndarray
        Apparent reflectance (ARHO_SIM) as observed by sensor

    Notes
    -----
    Modified to handle ground based spectral measurements from FLOX data specifics.
    """
    rho = _compute_reflectance(x, sp, wvl)

    fluorescence = _compute_fluorescence(x, sp, wvl)

    # Calculate ARHO_SIM using quadratic formula
    arho_sim = rho + fluorescence / Lin

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


#################################################################################

def _l2b_regularized_cost_function_optimization(
    wvl,
    apparent_reflectance,
    xa_mean,
    atm_func,
    sa,
    sy,
    lmb,
    l2b_wavelength_grid,
    max_iter=15,
):
    
    # Define spline knots for surface reflectance
    knots = np.array([
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
        712.1000,
        719.6833,
        727.2667,
        734.6333,
        741.6167,
        747.8333,
        755.5000,
        768.000,
        wvl[-1],
        wvl[-1],
        wvl[-1],
        wvl[-1]
    ])

    # Initialize B-spline for surface reflectance
    sp = BSpline(knots, xa_mean[8:], 3)

    # Set initial spline coefficients from prior mean
    sp.c = xa_mean[8:] 

    # Define cost function
    def cost_function(fx, x):
        c = (apparent_reflectance - fx).T @ sy @ (apparent_reflectance - fx) + g * (
            x - xa_mean
        ).T @ sa @ (x - xa_mean)
        return c

    # Define forward model and its Jacobian
    def fw(x):
        return l2b_forward_model(
            x,
            wvl,
            sp,
            Lin=atm_func["Lin"],
        )

    def jac(x):
        return _jacobian_calculation(fw, x)


    # --- Initialization ---
    y   = apparent_reflectance
    x0  = xa_mean.copy()   # initial state vector
    x1  = xa_mean.copy()   # initial state vector + 1 iteration
    n_x = x0.size          # number of state variables    
    c_ex1 = 20
    c_ex2 = 20
    c_ex3 = 20        
    D = np.eye(sa.shape[0]) # Identity matrix

    # Initial forward model evaluation
    fx0 = fw(x0)
    k0 = jac(x0)
    
    # initialization debug arrays
    x_iter       = np.full((n_x, max_iter + 1), np.nan)
    c1y_iter     = np.full((max_iter + 1), np.nan)
    c1x_iter     = np.full((max_iter + 1), np.nan)
    lmb2_iter    = np.full((max_iter + 1), np.nan)
    c_ex1_iter   = np.full((max_iter + 1), np.nan)
    c_ex2_iter   = np.full((max_iter + 1), np.nan)
    c_ex3_iter   = np.full((max_iter + 1), np.nan)
    LMgamma_iter = np.full((max_iter + 1), np.nan)
    fx_iter      = np.full((fx0.size, max_iter + 1), np.nan)
    A_iter       = np.full((n_x, n_x, max_iter + 1), np.nan)
    K_iter       = np.full((wvl.size, n_x, max_iter +1 ), np.nan)
    flag         = np.full((max_iter + 1), np.nan)


    # COST FUNCTION
    c0y = (y - fx0).T @ sy @ (y - fx0)
    c0x = (x0 - xa_mean).T @ (lmb * sa) @ (x0 - xa_mean)
    c0 = c0y + c0x

    # Assign first iteration values
    x_iter[:, 0]       = x0
    c1y_iter[0]     = c0y
    c1x_iter[0]     = c0x
    lmb2_iter[0]    = 0
    LMgamma_iter[0] = lmb
    fx_iter[:, 0]      = fx0
    flag[0]         = 1


    # First rough estimation
    idx_free = np.array([0, 3, 4] + list(range(8, n_x)))  

    for iter in range(2):
        # Reduce problem to free variables
        k0_redu = k0[:, idx_free]
        sa_redu = sa[np.ix_(idx_free, idx_free)]
        x0_redu = x0[idx_free]

        # Non-regularized update
        Ci  = k0_redu.T @ sy @ k0_redu
        rhs = k0_redu.T @ sy @ (y - fx0)
        dx_redu = np.linalg.solve(Ci, rhs)
        # dx_redu = np.linalg.lstsq(Ci, rhs, rcond=None)[0] # If the matrix is ill-conditioned and you prefer a least-squares solution:
        x1[idx_free] = x0_redu + dx_redu

        # Update Jacobian
        k1 = jac(x1)
        k1_redu = k1[:, idx_free]

        sx_redu = np.linalg.inv(k1_redu.T @ sy @ k1_redu)
        dx = xa_mean[idx_free] - x1[idx_free]
        den = dx.T @ sa_redu @ sx_redu @ sa_redu @ dx
        lmb_ECM = np.sqrt(len(idx_free) / den)

        x1_reg = np.linalg.solve((np.linalg.inv(sx_redu) + lmb_ECM * sa_redu),
                                (np.linalg.inv(sx_redu) @ x1[idx_free] + lmb_ECM * sa_redu @ xa_mean[idx_free]))

        # Update free parameters
        x1[idx_free] = x1_reg

        # Evaluate model with new x1 and Jacobian
        fx1_reg = fw(x1)
        k1_reg  = jac(x1)

        # Prepare for next iteration
        x0 = x1.copy()
        fx0 = fx1_reg.copy()
        k0 = k1_reg.copy()

    
    # Reset for full iteration
    xa = x1.copy()
    x0 = xa_mean.copy()
    x1 = x0.copy()
    fx0 = fw(x0)
    k0 = jac(x0)


    # Full iteration loop
    i = -1
    LMgamma = 1e-3 * np.max(np.diag(k0.T @ sy @ k0))
    
    while (c_ex1 >= 1e-3 or c_ex2 >= 1e-3 or c_ex3 >= 1e-6) and i < max_iter:
        i += 1
        Ci = k0.T @ sy @ k0 + LMgamma * D
        x1 = x0 + np.linalg.solve(Ci, k0.T @ sy @ (y - fx0))

        # Jacobian
        k0 = jac(x1)

        # ECM
        sx = np.linalg.inv(LMgamma * D + k0.T @ sy @ k0)
        dx = xa - x1
        den = dx.T @ sa @ sx @ sa @ dx
        if i <= 4:
            lmb_ECM = np.sqrt(n_x / den)
        lmb2_iter[i] = lmb_ECM

        x1_reg = np.linalg.solve((np.linalg.inv(sx) + lmb_ECM * sa),
                                (np.linalg.inv(sx) @ x1 + lmb_ECM * sa @ xa))
        #x1_reg = np.linalg.inv(np.linalg.inv(sx) + lmb_ECM * sa) @ (np.linalg.inv(sx) @ x1 + lmb_ECM * sa @ xa)

        fx1_reg = fw(x1_reg)
        k1_reg = jac(x1_reg)

        # Cost function
        c1y = (y - fx1_reg).T @ sy @ (y - fx1_reg)
        c1x = (x1_reg - xa).T @ (lmb_ECM * sa) @ (x1_reg - xa)
        c1 = c1y + c1x

        # Debug arrays
        x_iter[:, i] = x1_reg
        fx_iter[:, i] = fx1_reg
        c1y_iter[i] = c1y
        c1x_iter[i] = c1x


        # Updates LMgamma
        # LMgamma_method = 0  # Presse et al., x10
        # LMgamma_method = 1  # TrustRegion
        LMgamma_method = 2    # Geodesic

        if LMgamma_method == 1:  # TrustRegion
            if i == 1:
                increase_count = 0

            rho = compute_LM_gain_ratio(x0, x1, fx0, y, c0, c1, LMgamma, D, k0, sy)

            LMgamma, x1_reg, c1, fx1_reg, k1_reg, flag[i], skipIteration = LMgamma_update_TrustRegion(
                rho, LMgamma, x0, x1_reg, c0, c1, fx0, fx1_reg, k0, k1_reg, increase_count
            )

        elif LMgamma_method == 2:  # Geodesic
            rho = compute_LM_gain_ratio(x0, x1, fx0, y, c0, c1, LMgamma, D, k0, sy)

            LMgamma, x1_reg, c1, fx1_reg, k1_reg, flag[i], skipIteration = LMgamma_update_Geodesic(
                rho, LMgamma, x0, x1_reg, c0, c1, fx0, fx1_reg, k0, k1_reg
            )

        else:  # Press et al.
            LMgamma, x1_reg, c1, fx1_reg, k1_reg, flag[i], skipIteration = LMgamma_update(
                LMgamma, x0, x1_reg, c0, c1, fx0, fx1_reg, k0, k1_reg
            )


        # LMgamma update (placeholder)
        LMgamma_iter[i] = LMgamma

        # Posterior (only for debug)
        sx = np.linalg.inv(lmb_ECM * sa + k1_reg.T @ sy @ k1_reg)
        A_iter[:, :, i] = sx @ k1_reg.T @ sy @ k1_reg
        K_iter[:, :, i] = k1_reg

        # Convergence criteria
        c_ex1 = np.linalg.norm(k1_reg.T @ (fx1_reg - fx0), 2) / (1 + c1)
        c_ex2 = np.linalg.norm(x1_reg - x0, 2) / (1 + np.linalg.norm(x1_reg, 2))
        c_ex3 = abs(c1 - c0) / (1 + c1)

        c_ex1_iter[i] = c_ex1
        c_ex2_iter[i] = c_ex2
        c_ex3_iter[i] = c_ex3

        # Update for next iteration
        x0 = x1_reg.copy()
        fx0 = fx1_reg.copy()
        k0 = k1_reg.copy()
        c0 = c1


    sp.c = x1_reg[8:]
    reflectance = sp(l2b_wavelength_grid)
    sif = _sif_forward_model(x1_reg[:8].flatten(), l2b_wavelength_grid)


    # Compute posterior uncertainty
    sif_unc = MC_SIF_uncertainty_estimation(x1_reg, sx, l2b_wavelength_grid)


    return reflectance, sif, sif_unc



def compute_LM_gain_ratio(x0, x1, fx0, y, c0, c1, LMgamma, D, k0, Sy):
    """
    Computes the gain ratio (rho) for Levenberg-Marquardt update.

    Authors:
        Prof. Sergio Cogliati
        Dr. Pietro Chierichetti
        University of Milano-Bicocca
        Department of Earth and Environmental Sciences (DISAT)
        Email: sergio.cogliati@unimib.it

    Inputs:
        x0       - Current parameter vector
        x1       - Proposed updated parameter vector
        fx0      - Model output at x0
        y        - Observed data
        c0       - Cost at x0
        c1       - Cost at x1
        LMgamma  - Damping parameter
        D        - Damping matrix
        k0       - Jacobian matrix at x0
        Sy       - Weighting matrix for observations

    Output:
        rho      - Gain ratio used to evaluate the quality of the update
    """

    LHS = x1 - x0
    J = k0

    # Predicted cost reduction
    pred_red = 0.5 * (LHS.T @ (LMgamma * D @ LHS + J.T @ Sy @ (y - fx0)))

    # Avoid division by very small values
    if abs(pred_red) < 1e-12:
        pred_red = np.sign(pred_red) * 1e-12

    # Compute gain ratio
    rho = (c0 - c1) / pred_red

    # Clamp rho to avoid extreme values
    rho = max(min(rho, 10), -1)

    return rho


def LMgamma_update_Geodesic(rho, LMgamma, x0, x1_reg, c0, c1, fx0, fx1_reg, k0, k1_reg):
    """
    Updates LMgamma based on gain ratio rho in geodesic root square.

    Authors:
        Prof. Sergio Cogliati
        Dr. Pietro Chierichetti
        University of Milano-Bicocca
        Department of Earth and Environmental Sciences (DISAT)
        Email: sergio.cogliati@unimib.it

    Inputs:
        rho        - Gain ratio
        LMgamma    - Current damping parameter
        x0         - Current parameter vector
        x1_reg     - Previous value of x1_reg (preserved if update accepted)
        c0         - Current cost
        c1         - Previous value of c1 (preserved if update accepted)
        fx0        - Model output at x0
        fx1_reg    - Previous value of fx1_reg
        k0         - Jacobian at x0
        k1_reg     - Previous value of k1_reg

    Outputs:
        LMgamma    - Updated damping parameter
        x1_reg     - Reset parameter vector if update rejected
        c1         - Reset cost if update rejected
        fx1_reg    - Reset model output if update rejected
        k1_reg     - Reset Jacobian if update rejected
        flag       - Update status (1 accepted, 0 rejected)
        skipIteration - Boolean indicating if iteration should be skipped
    """

    if rho > 0:
        LMgamma = LMgamma / (1 + rho)
        flag = 1
        skipIteration = 0
    else:
        LMgamma = LMgamma * 2
        x1_reg = x0
        c1 = c0
        fx1_reg = fx0
        k1_reg = k0
        flag = 0
        skipIteration = 1

    # Numerical safety: clip LMgamma within bounds
    LMgamma = max(min(LMgamma, 1e10), 1e-10)

    return LMgamma, x1_reg, c1, fx1_reg, k1_reg, flag, skipIteration


def MC_SIF_uncertainty_estimation(x, sx, wvl):
    """
    Monte-Carlo approach for uncertainty estimation

    Args:
        x (np.ndarray): parameters
        sx (np.ndarray): retrieval covariance matrix
        wvl (np.ndarray): wavelengths

    Returns:
        np.ndarray: standard deviation of SIF across the Monte Carlo runs at each wavelength
    """

    def std_keep_95(run_matrix):
        m = np.array(run_matrix, dtype=float)
        low = np.nanpercentile(m, 2.5, axis=1, keepdims=True)
        high = np.nanpercentile(m, 97.5, axis=1, keepdims=True)
        mask = (m >= low) & (m <= high)
        filtered = np.where(mask, m, np.nan)
        return np.nanstd(filtered, axis=1)
    

    # nearest PSD
    def make_psd(matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < 0] = 0
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


    # mu = first 8 elements of x
    mu = x[:8].astype(float)

    # Round Sx to 6 decimals
    Sxx = np.round(sx, 6)

    # Extract top-left 8x8 submatrix
    Sigma = Sxx[:8, :8]

    Sigma = make_psd(Sigma)


    # Set random seed for reproducibility
    np.random.seed(0)
    R = np.random.multivariate_normal(mean=mu, cov=Sigma, size=300)

    # For each sample row, build a SIF spectrum
    n_wvl = wvl.shape[0]
    run_matrix = np.empty((n_wvl, R.shape[0]), dtype=float)
    run_matrix[:] = np.nan
    
    for mc_run  in range(R.shape[0]):
        try:
            run_matrix[:, mc_run ] = _compute_fluorescence(R[mc_run , :], np.nan, wvl)
        except Exception:
            # Assegna NaN alla colonna
            run_matrix[:, mc_run ] = np.nan

    # Standard deviation across runs (axis=1 for each wavelength)
    std_dev = std_keep_95(run_matrix)

    return std_dev

######################################################################



def _jacobian_calculation(func, x, epsilon=np.float64(1e-6)):
    num_parameters = len(x)
    x_perturbed = np.tile(x, (num_parameters + 1, 1)) + np.concatenate(
        [epsilon * np.eye(num_parameters), np.zeros((1, num_parameters))]
    )
    y = func(x_perturbed)
    return ((y[:-1, :] - y[-1, :]) * (1 / epsilon)).T



def _l2b_regularized_cost_function_optimization_OLD(
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
    
    # Define spline knots for surface reflectance
    knots = np.array([
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
        712.1000,
        719.6833,
        727.2667,
        734.6333,
        741.6167,
        747.8333,
        755.5000,
        768.000,
        wvl[-1],
        wvl[-1],
        wvl[-1],
        wvl[-1]
    ])

    # Initialize B-spline for surface reflectance
    sp = BSpline(knots, xa_mean[8:], 3)

    # Set initial spline coefficients from prior mean
    sp.c = xa_mean[8:] 

    # Define cost function
    def cost_function(fx, x):
        c = (apparent_reflectance - fx).T @ sy @ (apparent_reflectance - fx) + g * (
            x - xa_mean
        ).T @ sa @ (x - xa_mean)
        return c

    # Define forward model and its Jacobian
    def fw(x):
        return l2b_forward_model(
            x,
            wvl,
            sp,
            Lin=atm_func["Lin"],
        )

    def jac(x):
        return _jacobian_calculation(fw, x)

    # Levenberg-Marquardt optimization
    lm_gamma = 0.1

    # Initialization
    y = apparent_reflectance
    x0 = xa_mean.copy()
    x1 = xa_mean.copy()

    # Initial forward model evaluation
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
                    (np.sqrt(lm_gamma) * np.linalg.norm(np.eye(n_x) * (x1 - x0)))
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