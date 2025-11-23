import numpy as np
import os
from scipy.io import loadmat
from src.apply_SFM_FLOX_cov_sep import apply_SFM_FLOX_cov_sep_IPF


def FLOX_processing(
    inc_fluo,
    ref_fluo,
    wvl,
    uncertainty=None,
    cov=None
):
    """Method to compute apparent reflectance and call SIF retrieval function for FLOX data.

    Args:
        inc_FLUO (np.ndarray): Incident radiance in mW m^-2 sr^-1 nm^-1
        ref_FLUO (np.ndarray): Reflected radiance in mW m^-2 sr^-1 nm^-1
        wvl (np.ndarray): Wavelength vector in nm
        DOYdayfrac (np.ndarray): Day of year + fractional day
        UTC_time (list[str]): Timestamps
        uncertainty (str, optional): Path to .mat file that contains the uncertainties. Defaults to None.
        cov (str, optional): Path to .mat file that contains the inverse covariances. Defaults to None.


    Returns:
        list[np.ndarray]: 
            sif_array               # SIF spectrum computed using SFM method
            ref_array               # Reflectance spectrum computed using SFM method
            sif_array_u             # SIF spectrum without SFM method
            ref_array_u             # Reflectance spectrum without SFM method
            wvl_out                 # Output wavelength array
            sif_red_peak            # Maximum SIF at red peak
            sif_red_peak_wl         # Wavelength of maximum SIF at red peak
            sif_o2b_band            # SIF at O₂-B absorption band
            sif_farred_peak         # Maximum SIF at far-red peak
            sif_farred_peak_wl      # Wavelength of maximum SIF at far-red peak
            sif_o2a_band            # SIF at O₂-A absorption band
            sif_integrated          # Spectrally integrated SIF
            sif_o2a_uncertainty     # Uncertainty of SIF at O₂-A band
            sif_o2b_uncertainty     # Uncertainty of SIF at O₂-B band
    """

    # Compute apparent reflectance
    with np.errstate(divide='ignore', invalid='ignore'):
        app_ref = ref_fluo / inc_fluo
        app_ref[np.isnan(app_ref)] = 0.0
        app_ref[np.isinf(app_ref)] = 0.0

    # Initialize uncertainties
    flox_unc_inc_est = None
    flox_unc_ref_est = None
    app_ref_un_est = None

    # Load uncertainties
    if uncertainty is not None and os.path.isfile(uncertainty):
        # Load the .mat file, which should contain "wl_unc", "L_down_unc", "L_up_unc"
        unc_data = loadmat(uncertainty)
        if "wl_unc" in unc_data and "L_down_unc" in unc_data and "L_up_unc" in unc_data:
            wl_unc = unc_data["wl_unc"].flatten()
            L_down_unc = unc_data["L_down_unc"].flatten()
            L_up_unc = unc_data["L_up_unc"].flatten()

            # Interpolate only valid ranges [0,5]
            mask_down = (L_down_unc >= 0) & (L_down_unc < 5)
            mask_up = (L_up_unc >= 0) & (L_up_unc < 5)

            wl_unc_down = wl_unc[mask_down]
            val_down = L_down_unc[mask_down]

            wl_unc_up = wl_unc[mask_up]
            val_up = L_up_unc[mask_up]

            flox_unc_inc = np.interp(wvl, wl_unc_down, val_down, left=np.nan, right=np.nan) \
                if wl_unc_down.size > 1 else np.full_like(wvl, np.nan)
            flox_unc_ref = np.interp(wvl, wl_unc_up,   val_up,   left=np.nan, right=np.nan) \
                if wl_unc_up.size > 1 else np.full_like(wvl, np.nan)

            # Expand to 2D
            flox_unc_inc_est = flox_unc_inc[:, np.newaxis] * inc_fluo
            flox_unc_ref_est = flox_unc_ref[:, np.newaxis] * ref_fluo

            # Apparent reflectance uncertainty
            with np.errstate(divide='ignore', invalid='ignore'):
                part1 = (flox_unc_ref_est / inc_fluo)**2
                part2 = ((ref_fluo * flox_unc_inc_est)**2) / (inc_fluo**4)
                app_ref_un_est = np.sqrt(part1 + part2)
                app_ref_un_est[np.isnan(app_ref_un_est)] = 0.0
        else:
            print(
                f"Warning: {uncertainty} missing required variables (wl_unc, L_down_unc, L_up_unc).")
    else:
        print("No 'uncertainty' file provided or file not found. Skipping uncertainty steps...")


    # Main SFM retrieval

    # If 'cov' is None or doesn't exist, apply_SFM_FLOX_cov_sep might skip advanced covariance
    if cov is not None and os.path.isfile(cov):
        aux_cov_path = cov
    else:
        aux_cov_path = None
        print("No 'cov' file provided or file not found. 'apply_SFM_FLOX_cov_sep' may skip advanced covariance...")

    # set app_ref_un_est to zero if uncertainties were not read
    if app_ref_un_est is None:
        app_ref_un_est = np.zeros_like(app_ref)


    # Perform SIF retrieval using FLEX-IPF algorithm adapted to FLOX data
    (
        sif_array,               # SIF spectrum computed using SFM method
        ref_array,               # Reflectance spectrum computed using SFM method
        sif_array_u,             # SIF spectrum without SFM method
        ref_array_u,             # Reflectance spectrum without SFM method
        wvl_out,                 # Output wavelength array
        sif_red_peak,            # Maximum SIF at red peak
        sif_red_peak_wl,         # Wavelength of maximum SIF at red peak
        sif_o2b_band,            # SIF at O₂-B absorption band
        sif_farred_peak,         # Maximum SIF at far-red peak
        sif_farred_peak_wl,      # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,            # SIF at O₂-A absorption band
        sif_integrated,          # Spectrally integrated SIF
        sif_o2a_uncertainty,     # Uncertainty of SIF at O₂-A band
        sif_o2b_uncertainty      # Uncertainty of SIF at O₂-B band
    ) = apply_SFM_FLOX_cov_sep_IPF(
        inc_fluo_corr=inc_fluo,      # Incident irradiance from Fluo spectrometer
        app_ref=app_ref,             # Apparent reflectance
        app_ref_un=app_ref_un_est,   # Uncertainty of apparent reflectance
        wvl=wvl,                     # Wavelength vector
        path_aux=aux_cov_path        # Path to covariance matrix file
    )

    # --- Return all processed outputs ---
    return (
        sif_array,               # SIF spectrum computed using SFM method
        ref_array,               # Reflectance spectrum computed using SFM method
        sif_array_u,             # SIF spectrum without SFM method
        ref_array_u,             # Reflectance spectrum without SFM method
        wvl_out,                 # Output wavelength array
        sif_red_peak,            # Maximum SIF at red peak
        sif_red_peak_wl,         # Wavelength of maximum SIF at red peak
        sif_o2b_band,            # SIF at O₂-B absorption band
        sif_farred_peak,         # Maximum SIF at far-red peak
        sif_farred_peak_wl,      # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,            # SIF at O₂-A absorption band
        sif_integrated,          # Spectrally integrated SIF
        sif_o2a_uncertainty,     # Uncertainty of SIF at O₂-A band
        sif_o2b_uncertainty      # Uncertainty of SIF at O₂-B band
    )

