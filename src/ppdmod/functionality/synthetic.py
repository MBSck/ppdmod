import os
import numpy as np

from astropy.io import fits
from shutil import copyfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from astropy.units import Quantity

from .data_prep import DataHandler
from .utils import make_delta_component, make_ring_component, make_fixed_params
from .fitting_utils import calculate_model
from .plotting_utils import write_data_to_ini

np.seterr(divide='ignore')

def loop_model_for_wavelengths(data: DataHandler) -> Quantity:
    """Loops a model for the input parameters and returns the slices of the
    wavelength

    Parameters
    ----------
    data: DataHandler

    Returns
    -------
    Tuple
    """
    data.wavelengths = data.readouts[0].get_wavelength_solution()
    total_flux, corr_flux, cphases = calculate_model(data.initial, data, debug=True)
    return total_flux, corr_flux, cphases

def _reformat_array(input_array: Quantity) -> Quantity:
    """Reformats array into vertical columns"""
    unit = input_array.unit
    arrays = np.hsplit(input_array, input_array.shape[1])
    return [[float(value.value) for value in array] for array in arrays]*unit

def _reformat_to_matisse_data(total_flux, corr_flux, cphases) -> Tuple[np.ndarray]:
    """Reformats the arrays to the MATISSE-pipeline form"""
    total_flux = _reformat_array(total_flux)
    corr_flux = _reformat_array(corr_flux)
    cphases = _reformat_array(cphases)
    return total_flux, corr_flux, cphases

def create_model(data: DataHandler, save_path: Optional[Path] = "") -> None:
    """Saves the model as a '.fits'-file

    Parameters
    ----------
    data: DataHandler
    """
    print("The input parameters for the synthetic datasets creation.")
    print("------------------------------------------------")
    print(data.initial)

    if save_path:
        synthetic_path = os.path.join(save_path, "synthetic")
    else:
        synthetic_path = "synthetic"

    if not os.path.exists(synthetic_path):
        os.makedirs(synthetic_path)
    this_instant = datetime.now()
    model_path = os.path.join(synthetic_path,
                              f"{this_instant.date()}_{this_instant.time()}_model_synthetic")
    os.makedirs(model_path)
    np.save(os.path.join(model_path, "theta.npy"), data.initial)

    print(f"They have been saved to {model_path}")
    print("------------------------------------------------")
    total_flux, corr_flux,\
        cphases  = _reformat_to_matisse_data(*loop_model_for_wavelengths(data))
    total_flux_err, corr_flux_err,\
        cphases_err = map(lambda x: x*np.random.uniform(0, 0.4, x.shape),
                          (total_flux, corr_flux, cphases))

    for i, fits_file in enumerate(data.fits_files):
        output_file = os.path.join(model_path, f"synthetic_fit_{i}.fits")
        copyfile(fits_file, output_file)
        total_flux, total_flux_err = map(lambda x: x[i:i+1]\
                                         if i != len(data.fits_files) else x[i:],\
                                         (total_flux, total_flux_err))
        corr_flux, corr_flux_err = map(lambda x: x[i*6:(i+1)*6]\
                                       if i != len(data.fits_files) else x[i*6:],\
                                       (corr_flux, corr_flux_err))
        cphases, cphases_err = map(lambda x: x[i*4:(i+1)*4]\
                                   if i != len(data.fits_files) else x[i*4:],\
                                   (total_flux, total_flux_err))
        with fits.open(output_file, mode="update") as hdul:
            hdul["oi_vis"].data["visamp"] = corr_flux.value
            hdul["oi_vis"].data["visamperr"] = corr_flux_err.value
            hdul["oi_t3"].data["t3phi"] = cphases.value
            hdul["oi_t3"].data["t3phierr"] = cphases_err.value
            hdul["oi_flux"].data["fluxdata"] = total_flux.value
            hdul["oi_flux"].data["fluxerr"] = total_flux_err.value

            print(f"Model for {fits_file} saved as a {output_file}-file"\
                  f" created and updated with model values")
    data.theta_max = ""
    write_data_to_ini(data, "", "", "", save_path=model_path)


if __name__ == "__main__":
    data_path = "../../../data/hd_142666_jozsef/nband"
    fits_files = ["HD_142666_2022-04-23T03_05_25_N_TARGET_FINALCAL_INT.fits"]
    fits_files = [os.path.join(data_path, file) for file in fits_files]
    save_path = "../../../assets/model_results"
    wavelengths = [12.0]
    data = DataHandler(fits_files, wavelengths)
    complete_ring = make_ring_component("inner_ring",
                                        [[0., 0.], [0., 0.], [0.1, 6.], [0., 0.]])
    delta_component = make_delta_component("star")
    data.add_model_component(delta_component)
    data.add_model_component(complete_ring)
    data.fixed_params = make_fixed_params(30, 512, 1500, 7900, 140, 19, 1024)
    data.geometric_priors = [[0.4, 1.], [0, 180]]
    # data.modulation_priors = [[0., 1.], [0, 360]]
    data.disc_priors = [[0., 1.], [0., 1.]]
    data.lnf_priors = [-10., 10.]
    data.mcmc = [50, 50, 100, 1e-4]
    data.zero_padding_order = 2
    data.tau_initial = 1.
    create_model(data, "../../../assets/")

