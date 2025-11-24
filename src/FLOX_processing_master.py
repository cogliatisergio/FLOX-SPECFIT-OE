import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
#from src.FLOX_processing import FLOX_processing
from src.FLOX_processing_parallel import FLOX_processing


def FLOX_processing_master(data_path, uncertainty, cov):
    """
    Method that orchestrates FLOX data retrieval from CSV files

    Args:
        data_path (str): Directory path where CSV data files are located
        uncertainty (str): Path to the .mat file containing the FLOX uncertainties
        cov (str): Path to the .mat file containing the inverse covariances
    """

    # Initialize log file
    proc_time_all = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    logfile_name = os.path.join(data_path, f"{proc_time_all}_logfile.txt")

    logf = open(logfile_name, "w", encoding="utf-8")
    logf.write(f"FLOX_processing_master started at {proc_time_all}\n")
    logf.flush()

    # Suppress warnings:
    import warnings
    warnings.filterwarnings("ignore")

    # Log the input arguments:
    logf.write(f"data_path   = {data_path}\n")
    logf.write(f"uncertainty = {uncertainty}\n")
    logf.write(f"cov         = {cov}\n")
    logf.flush()

    # Adjust variables before running the program
    wvlRet = [670, 780]

    # Search for input files
    l0_pattern = os.path.join(data_path, "**", "Incoming*FLUO*.csv")
    l_pattern = os.path.join(data_path, "**", "Reflected*FLUO*.csv")

    l0_list = sorted(glob.glob(l0_pattern, recursive=True))
    l_list = sorted(glob.glob(l_pattern,  recursive=True))

    # Basic check confirming that the number of files matches:
    if len(l0_list) != len(l_list):
        msg = (f"Warning: found {len(l0_list)} 'Incoming FLUO' CSVs but "
               f"{len(l_list)} 'Reflected FLUO' CSVs. They should match.")
        logf.write(msg + "\n")
        logf.flush()

    # Init variables
    allm_specfit = []
    allf_specfit = None         # SIF 
    allf_unc_specfit = None     # SIF uncertainty 
    allr_specfit = None         # Reflectance    
    allar_unc = None            # Apparent Reflectance uncertainty 

    # Loop over each pair of CSV files
    n_tables = len(l0_list)
    for i_pair in range(n_tables):
        logf.write(f"\nProcessing file {i_pair+1} of {n_tables}\n")
        logf.flush()

        # Read the Incoming CSV
        fname_l0 = l0_list[i_pair]
        logf.write(f"Incoming: {fname_l0}\n")
        data_l0 = pd.read_csv(fname_l0, sep=";", header=0)
        wvl_qepro = data_l0.iloc[:, 0].to_numpy()
        l0_table = data_l0.iloc[:, 1:].to_numpy()

        # Read the Reflected CSV
        fname_l = l_list[i_pair]
        logf.write(f"Reflected: {fname_l}\n")
        data_l = pd.read_csv(fname_l, sep=";", header=0)
        l_table = data_l.iloc[:, 1:].to_numpy()

        # Compute UTC_time for the header information
        utc_column = data_l0.columns[1:]
        utc_time = [header.strip() for header in utc_column]
        folder_date = os.path.basename(
            os.path.dirname(fname_l0))

        doy_day_frac = []
        utc_datime_str = []
        for time_str in utc_time:
            dt_obj = datetime.strptime(
                folder_date + time_str, "%y%m%d%H_%M_%S")
            day_of_year = (dt_obj - datetime(dt_obj.year, 1, 1)).days + 1
            frac_day = (dt_obj - datetime(dt_obj.year,
                        dt_obj.month, dt_obj.day)).seconds / 86400
            doy_day_frac.append(day_of_year + frac_day)
            utc_datime_str.append(dt_obj.strftime("%d-%b-%Y %H:%M:%S"))

        # Wavelength Definition
        lb = np.argmin(np.abs(wvl_qepro - wvlRet[0]))
        ub = np.argmin(np.abs(wvl_qepro - wvlRet[1]))

        # Spectral subset of input spectra to min_wvl - max_wvl range and convert to mW
        wvl_sub = wvl_qepro[lb:ub+1]
        l_in = l0_table[lb:ub+1, :] * 1e3
        l_up = l_table[lb:ub+1, :] * 1e3



        # === FLOX Data Processing ===
        (
            sif,                # Retrieved SIF spectrum (array)
            ref,                # Retrieved reflectance spectrum (array)
            fluo_unc,           # Uncertainty for SIF retrieval
            ref_unc,            # Uncertainty for reflectance retrieval
            wvl_out,            # Output wavelength grid

            # Key SIF metrics
            sif_r_max,          # Maximum SIF value in the red region
            sif_r_wl,           # Wavelength corresponding to red peak
            sif_o2b,            # SIF value at O2-B absorption band
            sif_fr_max,         # Maximum SIF value in the far-red region
            sif_fr_wl,          # Wavelength corresponding to far-red peak
            sif_o2a,            # SIF value at O2-A absorption band
            sif_int,            # Integrated SIF over the spectrum

            # Uncertainties for key bands
            sif_o2a_un,         # Uncertainty at O2-A band
            sif_o2b_un,         # Uncertainty at O2-B band

            app_ref_unc         # Apparent reflectance uncertainty
        ) = FLOX_processing(
            Lin_array=l_in,          # Incident radiance (Lin)
            Lref_array=l_up,         # Reflected radiance (Lup)
            wvl=wvl_sub,             # Wavelength subset for processing
            uncertainty=uncertainty, # Optional uncertainty file path
            cov=cov                  # Covariance matrix for retrieval
        )



        # Accumulate the results
        out_arr_local = []
        for idx_col in range(l0_table.shape[1]):
            row = [
                doy_day_frac[idx_col],
                utc_datime_str[idx_col],
                idx_col+1,
                sif_fr_max[idx_col],
                sif_fr_wl[idx_col],
                sif_r_max[idx_col],
                sif_r_wl[idx_col],
                sif_o2b[idx_col],
                sif_o2a[idx_col],
                sif_int[idx_col],
                sif_o2b_un[idx_col],
                sif_o2a_un[idx_col]
            ]
            out_arr_local.append(row)

        # Store in a list:
        allm_specfit.extend(out_arr_local)

        # Also accumulate the SIF/reflectance spectra
        if allf_specfit is None:
            allf_specfit = sif
            allf_unc_specfit = fluo_unc
            allr_specfit = ref
            allar_unc = app_ref_unc
            all_utc_datetime_str = utc_datime_str
        else:
            # Concatenate horizontally
            allf_specfit = np.concatenate([allf_specfit, sif], axis=1)
            allf_unc_specfit = np.concatenate(
                [allf_unc_specfit, fluo_unc], axis=1)
            allr_specfit = np.concatenate([allr_specfit, ref], axis=1)
            allar_unc = np.concatenate([allar_unc, app_ref_unc], axis=1)
            all_utc_datetime_str.extend(utc_datime_str)

        logf.write(f"Finished file {i_pair+1}.\n")
        logf.flush()

    # write output files
    out_header = [
        "DOYdayfrac",
        "UTC_datetime",
        "filenum",
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2B",
        "SIF_O2A",
        "SIF_int",
        "SIF_O2B_un",
        "SIF_O2A_un"]

    final_sif_params_name = os.path.join(
        data_path,
        f"{proc_time_all}_pySPECFIT-OE_SIF_metrics.txt"
    )
    write_csv_with_headers(final_sif_params_name, allm_specfit, out_header)

    col_headers = ["wvl"] + all_utc_datetime_str
    arr_f_specfit = np.column_stack([wvl_out, allf_specfit])
    final_sif_name = os.path.join(
        data_path, f"{proc_time_all}_pySPECFIT-OE_SIF_spectrum.txt")
    write_csv_with_headers(final_sif_name, arr_f_specfit, col_headers)

    arr_f_unc_specfit = np.column_stack([wvl_out, allf_unc_specfit])
    final_sif_unc_name = os.path.join(
        data_path, f"{proc_time_all}_pySPECFIT-OE_SIF_spectrum_uncertainty.txt")
    write_csv_with_headers(
        final_sif_unc_name, arr_f_unc_specfit, col_headers)

    arr_r_specfit = np.column_stack([wvl_out, allr_specfit])
    final_r_name = os.path.join(
        data_path, f"{proc_time_all}_pySPECFIT-OE_REFLECTANCE_spectrum.txt")
    write_csv_with_headers(final_r_name, arr_r_specfit, col_headers)

    arr_ar_unc = np.column_stack([wvl_out, allar_unc])
    final_r_name = os.path.join(
        data_path, f"{proc_time_all}_pySPECFIT-OE_APPARENT_REFLECTANCE_UNCERTAINTY.txt")
    write_csv_with_headers(final_r_name, arr_ar_unc, col_headers)
    

    write_results_to_netcdf(
        output_path=data_path,
        proc_time=proc_time_all,
        wavelengths=wvl_out,
        timestamps=all_utc_datetime_str,
        sif_spectrum=allf_specfit,
        sif_uncertainty=allf_unc_specfit,
        reflectance_spectrum=allr_specfit,
        apparent_reflectance_uncertainty=allar_unc,
        sif_metrics=allm_specfit
    )

    # Close log file
    logf.write("\nFLOX_processing_master completed.\n")
    logf.close()


def write_csv_with_headers(filename, data2d, headers):
    """
    Utility method to write the CSV files.

    Args:
        filename (str): path to output CSV
        data2d (np.ndarray): 2D NumPy array to write
        headers (list): list of column headers (strings), length matches data2d.shape[1]
    """
    out_table = pd.DataFrame(data2d, columns=headers)
    out_table.to_csv(filename, sep=";", index=False,
                     na_rep="NaN", float_format="%.6f")
    



def write_results_to_netcdf(
    output_path,
    proc_time,
    wavelengths,
    timestamps,
    sif_spectrum,
    sif_uncertainty,
    reflectance_spectrum,
    apparent_reflectance_uncertainty,
    sif_metrics
):
    """
    Write FLOX processing results to a NetCDF file.

    Parameters
    ----------
    output_path : str
        Directory where the NetCDF file will be saved.
    proc_time : str
        Processing timestamp for naming and metadata.
    wavelengths : np.ndarray
        Wavelength grid (1D array).
    timestamps : list[str]
        List of UTC timestamps for each spectrum.
    sif_spectrum : np.ndarray
        SIF spectrum (2D array: wavelength x timestamp).
    sif_uncertainty : np.ndarray
        SIF uncertainty spectrum (same shape as sif_spectrum).
    reflectance_spectrum : np.ndarray
        Reflectance spectrum (same shape as sif_spectrum).
    reflectance_uncertainty : np.ndarray
        Apparent reflectance uncertainty (same shape as sif_spectrum).
    sif_metrics : dict
        Dictionary of SIF metrics:
        {
            "SIF_FARRED_max": float,
            "SIF_FARRED_max_wvl": float,
            "SIF_RED_max": float,
            "SIF_RED_max_wvl": float,
            "SIF_O2B": float,
            "SIF_O2A": float,
            "SIF_int": float,
            "SIF_O2B_un": float,
            "SIF_O2A_un": float
        }
    """
    netcdf_file = os.path.join(output_path, f"{proc_time}_pySPECFIT-OE_results.nc")

    metric_names = [
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2B",
        "SIF_O2A",
        "SIF_int",
        "SIF_O2B_un",
        "SIF_O2A_un"
    ]

    with Dataset(netcdf_file, "w", format="NETCDF4") as nc:
        # === Dimensions ===
        nc.createDimension("wavelength", len(wavelengths))
        nc.createDimension("timestamp", len(timestamps))

        # === Variables ===
        wvl_var = nc.createVariable("wavelength", "f4", ("wavelength",))
        wvl_var.units = "nm"
        wvl_var[:] = wavelengths

        time_var = nc.createVariable("timestamp", "S1", ("timestamp",))
        time_var[:] = np.array(timestamps, dtype="S")

        sif_var = nc.createVariable("SIF_spectrum", "f4", ("wavelength", "timestamp"), zlib=True)
        sif_var.units = "mW m-2 sr-1 nm-1"
        sif_var[:, :] = sif_spectrum

        sif_unc_var = nc.createVariable("SIF_uncertainty", "f4", ("wavelength", "timestamp"), zlib=True)
        sif_unc_var.units = "mW m-2 sr-1 nm-1"
        sif_unc_var[:, :] = sif_uncertainty

        refl_var = nc.createVariable("reflectance_spectrum", "f4", ("wavelength", "timestamp"), zlib=True)
        refl_var.units = "[-]"
        refl_var[:, :] = reflectance_spectrum

        refl_unc_var = nc.createVariable("apparent_reflectance_uncertainty", "f4", ("wavelength", "timestamp"), zlib=True)
        refl_unc_var.units = "[-]"
        refl_unc_var[:, :] = apparent_reflectance_uncertainty

        # === SIF metrics ===
        if isinstance(sif_metrics, list):
            metrics_array = np.array(sif_metrics)  # shape (n_timestamps, n_metrics)
            for i, name in enumerate(metric_names):
                var = nc.createVariable(name, "f4", ("timestamp",), zlib=True)





