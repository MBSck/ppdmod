from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from dynesty import DynamicNestedSampler, NestedSampler

from .basic_components import get_component_by_name
from .component import Component
from .options import OPTIONS, STANDARD_PARAMETERS
from .parameter import Parameter

# TODO: Include showing dynamic fitting
KEYWORD_DESCRIPTIONS = {
    "rchisq": "The total reduced chi square of the fit",
    "rchisqf": "The flux's reduced chi square of the fit",
    "rchisqv": "The visibility's reduced chi square of the fit",
    "rchisqc": "The closure phase's reduced chi square of the fit",
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

    # TODO: Make this work with the SED fit as well
    # rchi_sq_tot, rchi_sq_flux, rchi_sq_vis, rchi_sq_cp = compute_observable_chi_sq(
    #     *compute_observables(components), reduced=True, split=True)
    # header["RCHISQ"] = (np.round(rchi_sq_tot, 2), KEYWORD_DESCRIPTIONS["rchisq"])
    # header["RCHISQF"] = (np.round(rchi_sq_flux, 2), KEYWORD_DESCRIPTIONS["rchisqf"])
    # header["RCHISQV"] = (np.round(rchi_sq_vis, 2), KEYWORD_DESCRIPTIONS["rchisqv"])
    # header["RCHISQC"] = (np.round(rchi_sq_cp, 2), KEYWORD_DESCRIPTIONS["rchisqc"])
    # header["QUANTILES"[:8]] = (OPTIONS.fit.quantiles, KEYWORD_DESCRIPTIONS["quantiles"])
    # header["WAVELENGTH"[:8]] = (OPTIONS.fit.wavelengths, KEYWORD_DESCRIPTIONS["wavelengths"])

    for key in OPTIONS.fit.data:
        key = "vis" if key == "vis2" else key
        description = f"The weight for the {key.upper()} data set"
        header[f"W{key.upper()}"] = (np.round(getattr(OPTIONS.fit.weights, key), 3), description)

    if fit_hyperparameters is not None:
        for key, value in fit_hyperparameters.items():
            if key in ["ptform", "lnprob"]:
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

        for parameter in component.get_params().values():
            if parameter.grid is None:
                grid = np.full(parameter().shape, np.nan)
            else:
                grid = parameter.grid

            data[parameter.shortname] = (grid, parameter().value)

        table = fits.BinTableHDU(
                Table(data=data), header=table_header,
                name=component_labels[index].upper().replace(" ", "_"))
        tables.append(table)

    hdu = fits.HDUList([primary, *tables])
    hdu.writeto(save_dir, overwrite=True)


def restore_from_fits(path: Path,
                      name: Optional[str] = "model.fits"
                      ) -> Tuple[List[str], List[Component], Optional[NestedSampler]]:
    """Retrieves the individual model components from a model (.fits)-file 
    as well as the component labels and the sampler used.
    """
    components, component_labels = [], []
    model_fits = path / name
    with fits.open(model_fits, "readonly") as hdul:
        for card in hdul:
            header = card.header
            if card.name == "PRIMARY":
                OPTIONS.model.gridtype = header["GRIDTYPE"]
                OPTIONS.model.output = header["OUTTYPE"]
                OPTIONS.fit.method = header["FITMETH"]
                continue

            component_labels.append(card.name)
            param_names, param_data = card.data.columns.names, card.data.tolist()
            param_grid = [value if not np.all(np.isnan(value)) else None
                          for value in param_data[0]]

            params = []
            for name, value, grid in zip(param_names, param_data[1], param_grid):
                param_name = name
                if (name[0] == "c" or name[0] == "s") and len(name) <= 2:
                    param_name = name[0]

                if param_name not in vars(STANDARD_PARAMETERS):
                    if "kappa" in param_name:
                        param = Parameter(**STANDARD_PARAMETERS.kappa_abs)
                    if "weight" in param_name:
                        param = Parameter(**STANDARD_PARAMETERS.cont_weight)
                else:
                    param = Parameter(**getattr(STANDARD_PARAMETERS, param_name))

                param.grid, param.value = grid, value
                param.shortname = param.name = name
                params.append(param)

            component = get_component_by_name(header["COMP"])(**dict(zip(param_names, params)))
            components.append(component)

    OPTIONS.model.components_and_params = [[label.lower(), component.get_params()]
                                            for label, component in zip(component_labels, components)]

    # TODO: Add here the other samplers and emcee
    sampler = DynamicNestedSampler.restore(path / "sampler.save")

    return component_labels, components, sampler
