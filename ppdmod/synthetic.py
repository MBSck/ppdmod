from pathlib import Path
from shutil import copyfile

import astropy.units as u
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from .data import ReadoutFits
from .model import Model
from .fitting import calculate_observables


def create_synthetic(model: Model, fits_file: Path,
                     pixel_size: u.mas, save_path: Path) -> None:
    """Creates and then saves a model synthetically for many pixels."""
    readout = ReadoutFits(fits_file)
    total_fluxes, correlated_fluxes, closure_phases = [], [], []
    total_fluxes_err, correlated_fluxes_err, closure_phases_err = [], [], []
    for wavelength in tqdm(readout.wavelength):
        fourier_transform = model.calculate_complex_visibility(wavelength=wavelength)
        total_flux, correlated_flux, closure_phase =\
                calculate_observables(
                        fourier_transform, readout.ucoord, readout.vcoord,
                        readout.u123coord, readout.v123coord, pixel_size, wavelength)
        total_flux_err, correlated_flux_err, closure_phase_err =\
                map(lambda x: np.abs(x*np.random.uniform(0, 0.4, x.shape)),
                    (total_flux, correlated_flux, closure_phase))
        total_fluxes.append(total_flux)
        total_fluxes_err.append(total_flux_err)
        correlated_fluxes.append(correlated_flux)
        correlated_fluxes_err.append(correlated_flux_err)
        closure_phases.append(closure_phase)
        closure_phases_err.append(closure_phase_err)

    correlated_fluxes, correlated_fluxes_err = map(lambda x: np.stack(x, axis=1),
                                                   [correlated_fluxes, correlated_fluxes_err])
    closure_phases, closure_phases_err = map(lambda x: np.stack(x, axis=1),
                                             [closure_phases, closure_phases_err])

    output_file = Path(save_path) / "synthetic.fits"
    copyfile(fits_file, output_file)
    with fits.open(output_file, "update") as hdul:
        hdul["oi_flux"].data["fluxdata"] = total_fluxes
        hdul["oi_flux"].data["fluxerr"] = total_fluxes_err
        hdul["oi_vis"].data["visamp"] = correlated_fluxes
        hdul["oi_vis"].data["visamperr"] = correlated_fluxes_err
        hdul["oi_t3"].data["t3phi"] = closure_phases
        hdul["oi_t3"].data["t3phierr"] = closure_phases_err
