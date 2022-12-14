#!/usr/bin/env python3

import os
import numpy as np

from glob import glob
from pathlib import Path
from astropy.io import fits
from shutil import copyfile
from typing import Any, Dict, List, Union, Optional

from ..models import CompoundModel
from .fourier import FFT
from .readout import ReadoutFits
from .baseClasses import Model
from .utils import progress_bar
from .fitting_utils import get_rndarr_from_bounds

# TODO: Make the model save the numbers of pixel and zero padding that was used
# for the calculation and some more edge data (Do this within the fits file)

def loop_model4wl(model, theta: List, bb_params: List,
                  mas_size: int, px_size: int,
                  fits_file: Path, zero_padding_order: Optional[int] = 1,
                  intp: Optional[bool] = True) -> np.ndarray:
    """Loops a model for the input parameters and returns the slices of the
    wavelength

    Parameters
    ----------
    model
    theta: List
    bb_params: List
    mas_size: int
    px_size: int
    fits_file: Path

    Returns
    -------
    """
    readout = ReadoutFits(fits_file)
    wl = readout.get_wl()
    uvcoords, t3phi_uvcoords = readout.get_uvcoords(),\
            readout.get_t3phi_uvcoords()

    amp_lst, amperr_lst  = [[[] for i in range(6)] for j in range(2)]
    phase_lst, phaseerr_lst = [[[] for i in range(4)] for j in range(2)]
    flux_lst, fluxerr_lst = [], []

    print(f"Polychromatic {model(*bb_params, 1).name} is being calculated!")
    progress_bar(0, len(wl))

    for i, o in enumerate(wl):
        mod = model(*bb_params, o)
        flux = mod.eval_model(theta, mas_size, px_size)
        total_flux = np.sum(flux)
        fft = FFT(flux, o, mod.pixel_scale, zero_padding_order)
        amp, phase, xycoords  = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                                intp, True)

        for j, l in enumerate(amp):
            amp_lst[j].append(l)
            amperr_lst[j].append((1/(o*1e6)*\
                                 np.random.uniform(0, np.max(l))).tolist())

        for j, l in enumerate(phase):
            phase_lst[j].append(l)
            phaseerr_lst[j].append((1/(o*1e6)*\
                                    np.random.uniform(0, np.max(l))).tolist())

        flux_lst.append(total_flux)
        fluxerr_lst.append((1/(o*1e6)*\
                            np.random.uniform(0, total_flux)).tolist())

        progress_bar(i + 1, len(wl))
    print()

    model_params_dict = {"theta": theta, "blackbody": bb_params,
                         "fov_size": mas_size, "npx": px_size,
                         "pixel_scale": mod.pixel_scale}

    return amp_lst, amperr_lst, phase_lst,\
            phaseerr_lst, flux_lst, fluxerr_lst, model_params_dict

def create_model(save_path: Path, model: Model,
                 theta: List, bb_params: List, fov_size: int,
                 pixel_size: int,
                 path_to_fits: Optional[Path] = "",
                 fits_file: Optional[Path] = "",
                 zero_padding_order: Optional[int] = 2,
                 intp: Optional[bool] = True) -> None:
    """Saves the model as a '.fits'-file

    Parameters
    ----------
    save_path: Path
    data: List
    model: Model
    theta: List
    bb_params: List
    fov_size: int
    pixel_size: int
    path_to_fits: Path, optional
    fits_file: Path, optional
    zero_padding_order: int, optional
    intp: bool, optional
    """
    print("The input parameters for the synthetic dataset's creation.")
    print("------------------------------------------------")
    print(theta)

    np.save(os.path.join('assets', 'theta.npy'), theta)
    print(f"They have been saved to {os.path.join('assets', 'theta.npy')}")
    print("------------------------------------------------")

    if path_to_fits:
        fits_files = glob(os.path.join(path_to_fits, "*.fits"))
    elif fits_file:
        fits_files = [fits_file]
    else:
        raise IOError("Either 'path_to_fits' or 'fits_file' must be set!")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        print(f"Created folder {os.path.dirname(save_path)}!")

    for number, fits_file in enumerate(fits_files):
        print(f"Model for {fits_file} is being calculated!")
        print("------------------------------------------------")

        output_file = f"{save_path}_{number}.fits"

        data = loop_model4wl(model, theta, bb_params, fov_size,
                             pixel_size, fits_file, zero_padding_order, intp)

        amp, amperr, phase, phaseerr,\
                flux, fluxerr= map(lambda x: np.array(x), data[:-1])
        model_params_dict = data[-1]

        copyfile(fits_file, output_file)

        with fits.open(output_file, mode="update") as hdul:
            hdul["oi_vis"].data["visamp"] = amp
            hdul["oi_vis"].data["visamperr"] = amperr
            hdul["oi_t3"].data["t3phi"] = phase
            hdul["oi_t3"].data["t3phierr"] = phaseerr
            hdul["oi_flux"].data["fluxdata"] = flux
            hdul["oi_flux"].data["fluxerr"] = fluxerr

#        card_lst = []
#        for i, o in model_params_dict.items():
#            card_lst.append(fits.Card(i, o))
#
#        hdr = fits.Header(card_lst)
#        hdul.append("oi_param")

        print(f"Model for {fits_file} saved as a {output_file}-file"\
              f" created and updated with model values")


if __name__ == "__main__":
    ...

