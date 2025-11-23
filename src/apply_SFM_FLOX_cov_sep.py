import numpy as np
import os
import h5py
from tqdm import tqdm
#from src.auxiliar.fg_sif_spectrum import fg_sif_spectrum
from src.auxiliar.numeric_jacobian import numeric_jacobian
#from src.auxiliar.specfit_tayor_l2a_8parms_func_prox_sensing import specfit_tayor_l2a_8parms_func_prox_sensing
#from src.auxiliar.sif_fwmodel_8parms import sif_fwmodel_8parms
from src.sif_retrieval_IPF_v2_1_modified import _l2b_regularized_cost_function_optimization


def sif_parms_flox(wl, sif, sif_un):
    """
    Extracts key SIF metrics (red peak, far-red peak,
    O2-B, O2-A, integrated SIF, etc.) from the retrieved fluorescence.
    Also includes uncertainties from sif_array_u

    Args:
        wl (np.ndarray): wavelengths
        sif (np.ndarray): solar induced fluorescence
        sif_un (np.ndarray): solar induced fluorescence uncertainties

    Returns:
        list[np.ndarray]: sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un
    """

    # Ensure inputs are numpy arrays
    wl = np.array(wl)
    sif = np.array(sif)
    sif_un = np.array(sif_un)

    # RED SIF
    # max
    red_indices = wl < 690
    sif_r = sif[red_indices, :]
    sif_r_max = np.max(sif_r, axis=0)
    id = np.argmax(sif_r, axis=0)
    sif_r_wl = wl[red_indices][id]

    # SIF at O2-B 687nm
    ii = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[ii, :]
    sif_o2b_un = sif_un[ii, :]

    # FAR-RED SIF
    far_red_indices = wl > 720
    sif_fr = sif[far_red_indices, :]
    sif_fr_max = np.max(sif_fr, axis=0)
    p = np.argmax(sif_fr, axis=0)
    x = wl[far_red_indices]
    sif_fr_wl = x[p]

    # SIF at O2-A 760nm
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii, :]
    sif_o2a_un = sif_un[ii, :]

    # Spectrally integrated SIF
    sif_int = np.trapezoid(sif, wl, axis=0)

    return (sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un)




def apply_SFM_FLOX_cov_sep_IPF(inc_fluo_corr, app_ref, app_ref_un, wvl, path_aux):
    """
    Core spectral fitting method for FLOX data (single-thread, Rodgers LM approach).

    Args:
        inc_FLUO_corr (np.ndarray): incident radiance
        app_ref (np.ndarray): apparent reflectance
        app_ref_un (np.ndarray): reflectance uncertainties
        wvl (np.ndarray): wavelength vector
        sifo2a (np.ndarray): initial guess arrays for O2-A SIF
        sifo2b (np.ndarray): initial guess arrays for O2-B SIF
        path_aux (str): path to .mat file with invCOV_SIF_RHO

    Returns:
        list[np.ndarray]: sif_array, ref_array, sif_array_u, ref_array_u, wv_out,
            sif_r_max, sif_r_wl, sif_o2b,
            sif_fr_max, sif_fr_wl, sif_o2a,
            sif_int, sif_o2a_un, sif_o2b_un
    """

####################################

    #  Load Inverse parameter covariance matrix
    if not os.path.isfile(path_aux):
        raise FileNotFoundError(f"Cannot find covariance file: {path_aux}")

    with h5py.File(path_aux, 'r') as f:
        if 'invCOV_SIF_RHO' not in f:
            raise KeyError(
                "invCOV_SIF_RHO not found in the provided netCDF file.")
        sa = np.array(f['invCOV_SIF_RHO']).astype(float)
        xa_mean = np.array(f['xa_mean']).astype(float).ravel()
        
    nparams = sa.shape[0]
    if nparams < 9:
        raise ValueError(
            "invCOV_SIF_RHO matrix expected >= 26x26. Found smaller.")    

    # Adjust covariance matrix values
    sa[8:, :8] = 1e-15
    sa[:8, 8:] = 1e-5
    sa[0, 3] = 1e-15
    sa[3, 0] = 1e-15
    sa[:2, 3:] = 1e-15
    sa[3:, :2] = 1e-15


    # Initialize output arrays
    n_wvl, n_spectra = inc_fluo_corr.shape


    # Undersample the wavelength grid for L2B products
    # Create independent copies to avoid modifying the original array
    l2b_wavelength_grid = np.copy(wvl)  # Wavelength grid for L2B processing
    wvl_out = np.copy(wvl)              # Output wavelength grid


    sif_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    sif_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)


    
    # Main loop over the different spectra
    #for num_spec in range(n_spectra):   

    # Main loop over the different spectra with progress bar
    for num_spec in tqdm(range(n_spectra), desc="Processing spectra", unit="spectra"):

        # Build data covariance (diagonal from app_ref_un^2)
        var_vec = (app_ref_un[:, num_spec])**2
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_var_vec = np.where(var_vec > 1e-30, 1.0/var_vec, 0.0)
        sy = np.diag(inv_var_vec)

        # Extract current spectra (Lin and apparent reflectance)
        l_incident = inc_fluo_corr[:, num_spec]
        atm_func = {"Lin": l_incident}
        farho = app_ref[:, num_spec]      


        lmb = 1e-4

        reflectance, sif, sif_unc = _l2b_regularized_cost_function_optimization(
            wvl,
            farho,
            xa_mean,
            atm_func,
            sa,
            sy,
            lmb,
            l2b_wavelength_grid,
            max_iter=15,
        )
        
        # store
        ref_array[:, num_spec] = reflectance
        sif_array[:, num_spec] = sif
        sif_array_u[:, num_spec] = sif_unc
        #ref_array_u[:, num_spec] = rho_unc
        #residual_sfm[:, num_spec] = residual



    # --- Compute SIF (Solar-Induced Fluorescence) Parameters ---
    (
        sif_red_peak,           # Maximum SIF at red peak
        sif_red_peak_wl,        # Wavelength of maximum SIF at red peak
        sif_o2b_band,           # SIF value at O2-B absorption band
        sif_farred_peak,        # Maximum SIF at far-red peak
        sif_farred_peak_wl,     # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,           # SIF value at O2-A absorption band
        sif_integrated,         # Spectrally integrated SIF over the full range
        sif_o2b_uncertainty,    # Uncertainty of SIF at O2-B band
        sif_o2a_uncertainty     # Uncertainty of SIF at O2-A band
    ) = sif_parms_flox(l2b_wavelength_grid, sif_array, sif_array_u)


    return (
        sif_array,               # SIF spectrum computed using SFM method
        ref_array,                # Reflectance spectrum computed using SFM method
        sif_array_u,            # SIF spectrum without SFM method
        ref_array_u,             # Reflectance spectrum without SFM method
        wvl_out,                # Output wavelength array
        sif_red_peak,           # Maximum SIF at red peak
        sif_red_peak_wl,        # Wavelength of maximum SIF at red peak
        sif_o2b_band,           # SIF at O2-B absorption band
        sif_farred_peak,        # Maximum SIF at far-red peak
        sif_farred_peak_wl,     # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,           # SIF at O2-A absorption band
        sif_integrated,         # Spectrally integrated SIF
        sif_o2a_uncertainty,    # Uncertainty of SIF at O2-A band
        sif_o2b_uncertainty     # Uncertainty of SIF at O2-B band
    )



'''
def apply_SFM_FLOX_cov_sep(inc_fluo_corr, app_ref, app_ref_un, wvl,
                           sifo2a, sifo2b, path_aux):
    """
    Core spectral fitting method for FLOX data (single-thread, Rodgers LM approach).

    Args:
        inc_FLUO_corr (np.ndarray): incident radiance
        app_ref (np.ndarray): apparent reflectance
        app_ref_un (np.ndarray): reflectance uncertainties
        wvl (np.ndarray): wavelength vector
        sifo2a (np.ndarray): initial guess arrays for O2-A SIF
        sifo2b (np.ndarray): initial guess arrays for O2-B SIF
        path_aux (str): path to .mat file with invCOV_SIF_RHO

    Returns:
        list[np.ndarray]: sif_array, ref_array, sif_array_u, ref_array_u, wv_out,
            sif_r_max, sif_r_wl, sif_o2b,
            sif_fr_max, sif_fr_wl, sif_o2a,
            sif_int, sif_o2a_un, sif_o2b_un
    """

    #  Load Inverse parameter covariance matrix
    if not os.path.isfile(path_aux):
        raise FileNotFoundError(f"Cannot find covariance file: {path_aux}")

    with h5py.File(path_aux, 'r') as f:
        if 'invCOV_SIF_RHO' not in f:
            raise KeyError(
                "invCOV_SIF_RHO not found in the provided mat file.")
        sa = np.array(f['invCOV_SIF_RHO']).astype(float)

    nparams = sa.shape[0]
    if nparams < 9:
        raise ValueError(
            "invCOV_SIF_RHO matrix expected >= 9x9. Found smaller.")

    sa[8:, :8] = 1e-5
    sa[:8, 8:] = 1e-5
    sa[0, 3] = 1e-15
    sa[3, 0] = 1e-15
    sa[6, :] = 1e7
    sa[:, 6] = 1e7

    g = 1e-3
    sa = sa * g

    # Exclude O2 region from reflectance for B-Spline fit
    mask_id0 = (wvl > 686) & (wvl < 692)   # O2-B
    mask_id1 = (wvl > 758) & (wvl < 773)   # O2-A
    wvl_no_abs = wvl[~(mask_id0 | mask_id1)]  # ignoring O2 lines

    # Build knots with repeated endpoints
    knots = np.concatenate([
        [wvl[0]]*4,
        [675.0, 682.6, 693.45, 695.45, 699.0333,
         704.45, 712.1, 719.6833, 727.2667, 734.6333,
         741.6167, 747.8333, 755.5, 768.0],
        [wvl[-1]]*4
    ])

    # Initialize output arrays
    n_wvl, n_spectra = inc_fluo_corr.shape
    wv_out = wvl.copy()

    sif_array = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    sif_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)

    residual_sfm = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    max_i = 10
    c_inspect = np.full((20, n_spectra), np.nan, dtype=float)
    xx1_inspect = np.full((20, n_spectra), np.nan, dtype=float)
    number_tot_ite = np.zeros(n_spectra, dtype=int)

    # Keep the initial guess for SIF parameters
    sif_fg = np.full((n_spectra, 8), np.nan, dtype=float)

    # Main loop
    for num_spec in range(n_spectra):
        # Build data covariance (diagonal from app_ref_un^2)
        var_vec = (app_ref_un[:, num_spec])**2
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_var_vec = np.where(var_vec > 1e-30, 1.0/var_vec, 0.0)
        sy = np.diag(inv_var_vec)

        # Extract local arrays
        l_incident = inc_fluo_corr[:, num_spec]
        farho = app_ref[:, num_spec]

        # Subset to remove O2 absorption
        farho_no_abs = farho[~(mask_id0 | mask_id1)]

        if len(farho_no_abs) != len(wvl_no_abs):
            print(
                f"Warning: mismatch in lengths for wvl_sub({len(wvl_no_abs)}) vs. farho_no_abs({len(farho_no_abs)})")

        # Fit the B-spline
        sp_fit = make_lsq_spline(wvl_no_abs, farho_no_abs, knots, k=3)
        # Get the fitted weight
        p_r = sp_fit.c

        # Build initial guess for SIF
        pos_o2b = np.argmin(np.abs(wvl - 687.2))
        pos_o2a = np.argmin(np.abs(wvl - 760.3))

        fg_sifo2b = sifo2b[pos_o2b, num_spec]
        fg_sifo2a = sifo2a[pos_o2a, num_spec]
        if np.isnan(fg_sifo2b):
            fg_sifo2b = 0.0
        if np.isnan(fg_sifo2a):
            fg_sifo2a = 0.0
        if fg_sifo2b < 0:
            fg_sifo2b = 0.0
        if fg_sifo2a < 0:
            fg_sifo2a = 0.0

        x_sif_fg, _ = fg_sif_spectrum(fg_sifo2b, fg_sifo2a, wvl)

        sif_fg[num_spec, :] = x_sif_fg

        x0 = np.concatenate([x_sif_fg, p_r], axis=0)

        # Rodgers LM approach
        w_vec = np.ones(n_wvl, dtype=float)

        def Fw(x_in):
            new_w = x_in[8:]
            # Assign the new weights to sp_fit
            sp_fit.c = new_w
            # Evaluate the forward model
            return specfit_tayor_l2a_8parms_func_prox_sensing(
                x_in, wvl, sp_fit, l_incident, w_vec
            )

        fx0 = Fw(x0)
        y = farho
        xa = x0.copy()

        def cost_func(x_in, fx_in):
            dcost = 0.5 * (y - fx_in).dot(sy.dot(y - fx_in))
            pcost = 0.5 * (x_in - xa).dot(sa.dot(x_in - xa))
            return dcost + pcost

        xx0 = cost_func(x0, fx0)
        lm_gamma = 1.0
        c = 20.0
        i_iter = 0

        k = numeric_jacobian(Fw, x0)

        while (c >= 1e-4) and (i_iter < max_i):
            i_iter += 1

            A = (1+lm_gamma)*sa + k.T @ sy @ k
            b = k.T @ sy @ (y - fx0 + k @ (x0 - xa))

            try:
                dx = np.linalg.inv(A) @ b
            except np.linalg.LinAlgError:
                dx = np.zeros_like(x0)

            x1 = xa + dx
            fx1 = Fw(x1)
            # xx1 equal up to 3 decimal cases xx0 up to 4 decimal cases
            xx1 = cost_func(x1, fx1)

            xx1_inspect[i_iter-1, num_spec] = xx1
            if xx1 > xx0:
                # revert
                x1 = x0
                fx1 = fx0
                xx1 = xx0
                lm_gamma *= 10.0
            else:
                lm_gamma /= 10.0

            # OK here
            c = np.sum((y - fx1)**2)/n_wvl * 100.0
            c_inspect[i_iter-1, num_spec] = c

            x0 = x1
            fx0 = fx1
            xx0 = xx1

        number_tot_ite[num_spec] = i_iter

        # Posterior error covariance
        a_final = sa/g + k.T @ sy @ k
        try:
            sx = np.linalg.inv(a_final)
        except np.linalg.LinAlgError:
            sx = np.zeros_like(a_final)

        # Evaluate final reflectance & SIF
        sp_fit.c = x0[8:]
        rho = sp_fit(wvl)

        # reflectance uncertainty
        param_unc = np.sqrt(np.diag(sx))
        # interpret that as reflectance uncertainty weights
        sp_fit.c = param_unc[8:]
        rho_unc = sp_fit(wvl)
        # revert
        sp_fit.c = x0[8:]

        # SIF
        sif = sif_fwmodel_8parms(x0[:8], wvl)

        # SIF uncertainty (Monte Carlo)
        fluo_unc = MC_SIF_uncertainty_estimation(x0, sx, wvl)

        # residual
        residual = y - fx0

        # store
        sif_array[:, num_spec] = sif
        ref_array[:, num_spec] = rho
        sif_array_u[:, num_spec] = fluo_unc
        ref_array_u[:, num_spec] = rho_unc
        residual_sfm[:, num_spec] = residual

    # Compute Final SIF Parameters
    (sif_r_max, sif_r_wl, sif_o2b,
     sif_fr_max, sif_fr_wl, sif_o2a,
     sif_int, sif_o2b_un, sif_o2a_un
     ) = sif_parms_flox(wvl, sif_array, sif_array_u)

    return (sif_array, ref_array, sif_array_u, ref_array_u, wv_out,
            sif_r_max, sif_r_wl, sif_o2b,
            sif_fr_max, sif_fr_wl, sif_o2a,
            sif_int, sif_o2a_un, sif_o2b_un)
'''



