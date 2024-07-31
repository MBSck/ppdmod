from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from dynesty import NestedSampler, DynamicNestedSampler

from .basic_components import get_component_by_name
from .component import Component
from .fitting import compute_observables, compute_observable_chi_sq
from .options import OPTIONS, STANDARD_PARAMETERS
from .parameter import Parameter


# TODO: Include showing dynamic fitting
KEYWORD_DESCRIPTIONS = {
    "chisqr": "The reduced chi square of the fit",
    "data": "The data sets used for fitting",
    "date": "Creation date",
    "discard": "Chain discarded until element for emcee",
    "fitmeth": "Method applied for fitting",
    "gridtype": "The type of the model grid",
    "nburnin": "Number of burnin-steps for emcee",
    "ncore": "Numbers of cores for the fitting",
    "nlive": "Number of live points for dynesty",
    "nlive_init": "Number of initial live points for dynesty",
    "nsteps": "Number of steps for emcee",
    "nwalkers": "Numbers of walkers for emcee",
    "object": "Name of the object",
    "outtype": "The output type of the model",
    "quantiles": "The quantiles used for the fitting",
    "wavelengths": "The wavelengths used for fitting",
    "weights": "The weights for the different data sets",
}


def save_fits(components: List[Component],
              component_labels: List[str],
              save_dir: Optional[Path] = None,
              object_name: Optional[str] = None,
              fit_hyperparameters: Optional[Dict] = None,
              ncores: Optional[int] = None) -> None:
    """Saves a (.fits)-file of the model with all the information on the
    parameter space."""
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)

    header = fits.Header()
    header["OBJECT"] = (object_name, KEYWORD_DESCRIPTIONS["object"])
    header["FITMETH"] = (OPTIONS.fit.method, KEYWORD_DESCRIPTIONS["fitmeth"])

    rchi_sq = compute_observable_chi_sq(
            *compute_observables(components), reduced=True)
    header["CHISQR"] = (np.round(rchi_sq, 2), KEYWORD_DESCRIPTIONS["chisqr"])
    # header["QUANTILES"[:8]] = (OPTIONS.fit.quantiles, KEYWORD_DESCRIPTIONS["quantiles"])
    # header["WAVELENGTH"[:8]] = (OPTIONS.fit.wavelengths, KEYWORD_DESCRIPTIONS["wavelengths"])

    for key in OPTIONS.fit.data:
        key = "vis" if key == "vis2" else key
        description = f"The weight for the {key.upper()} data set"
        header[f"W{key.upper()}"] = (np.round(getattr(OPTIONS.fit.weights, key), 3), description)

    if fit_hyperparameters is not None:
        for key, value in fit_hyperparameters.items():
            if key in ["ptform"]:
                continue
            header[key.upper()[:8]] = (value, KEYWORD_DESCRIPTIONS[key.lower()])

    header["NCORE"] = (ncores, KEYWORD_DESCRIPTIONS["ncore"])
    header["GRIDTYPE"] = (OPTIONS.model.gridtype, KEYWORD_DESCRIPTIONS["gridtype"])
    header["OUTTYPE"] = (OPTIONS.model.output, KEYWORD_DESCRIPTIONS["outtype"])
    header["DATE"] = (f"{datetime.now()}", KEYWORD_DESCRIPTIONS["date"])
    primary = fits.PrimaryHDU(header=header)

    tables = []
    for index, component in enumerate(components):
        table_header, data = fits.Header(), {}
        table_header["COMP"] = component.shortname

        # TODO: Also save if it is thin or asymmetric here?
        for parameter in component.get_params().values():

            wavelength = np.full(parameter().shape, np.nan) \
                if parameter.wavelength is None else parameter.wavelength
            data[parameter.shortname] = (wavelength, parameter().value)

        table = fits.BinTableHDU(
                Table(data=data), header=table_header,
                name=component_labels[index].upper().replace(" ", "_"))
        tables.append(table)

    hdu = fits.HDUList([primary, *tables])
    hdu.writeto(save_dir, overwrite=True)


def restore_from_fits(path: Path) -> Tuple[List[str], List[Component], Optional[NestedSampler]]:
    """Retrieves the individual model components from a model (.fits)-file 
    as well as the component labels and the sampler used.
    """
    components, component_labels = [], []
    model_fits = path / "model.fits"
    with fits.open(model_fits, "readonly") as hdul:
        for card in hdul:
            header = card.header
            if card.name == "PRIMARY":
                continue

            component_labels.append(card.name)
            param_names, param_data = card.data.columns.names, card.data.tolist()
            param_wavelength = [value if not np.all(np.isnan(value)) else None
                                for value in param_data[0]]

            params = []
            for name, wavelength, value in zip(param_names, param_wavelength, param_data[1]):
                param_name = name
                if (name[0] == "c" or name[0] == "s") and len(name) <= 2:
                    param_name = name[0]

                param = Parameter(**getattr(STANDARD_PARAMETERS, param_name))
                param.wavelength, param.value = wavelength, value
                param.shortname = param.name = name
                params.append(param)

            component = get_component_by_name(header["COMP"])(**dict(zip(param_names, params)))
            components.append(component)

    # TODO: Add here the other samplers and emcee
    sampler = DynamicNestedSampler.restore(path / "sampler.save")

    return component_labels, components, sampler
