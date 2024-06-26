from datetime import datetime
from typing import Optional, List
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .component import Component
from .options import OPTIONS


def save_fits(dim: int, pixel_size: u.mas,
              distance: u.pc,
              components: List[Component],
              component_labels: List[str],
              wavelength: Optional[u.Quantity[u.um]] = None,
              opacities: List[np.ndarray] = None,
              save_dir: Optional[Path] = None,
              make_plots: Optional[bool] = False,
              object_name: Optional[str] = None,
              nlive: Optional[int] = None,
              sample: Optional[str] = None,
              bound: Optional[str] = None,
              nwalkers: Optional[int] = None,
              nburnin: Optional[int] = None,
              nsteps: Optional[int] = None,
              ncores: Optional[int] = None, **kwargs) -> None:
    """Saves a (.fits)-file of the model with all the information on the
    parameter space."""
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    pixel_size = u.Quantity(pixel_size, u.mas)
    wavelength = u.Quantity(wavelength, u.um) if wavelength is not None\
        else OPTIONS.fit.wavelengths
    distance = u.Quantity(distance, u.pc)

    total_flux = np.empty(wavelength.size)*u.Jy
    image, tables = np.empty((wavelength.size, dim, dim)), []
    for index, component in enumerate(components):
        comp_dir = save_dir / component_labels[index].lower().replace(" ", "_")
        image += component.compute_image(dim, pixel_size, wavelength)

        table_header = fits.Header()
        table_header["COMP"] = component.shortname
        table_header["GRIDTYPE"] = (OPTIONS.model.gridtype,
                                    "The type of the model grid")

        data = {"wavelength": wavelength}
        if component.shortname not in ["Star", "Point"]:
            component.dim.value = dim
            radius = np.tile(component.compute_internal_grid(), (wavelength.size, 1))
            temperature = component.compute_temperature(radius)
            surface_density = component.compute_surface_density(radius)
            optical_depth = component.compute_optical_depth(
                radius, wavelength[:, np.newaxis])
            emissivity = component.compute_emissivity(
                    radius, wavelength[:, np.newaxis])
            brightness = component.compute_intensity(
                    radius, wavelength[:, np.newaxis])

            data["radius"] = radius
            data["temperature"] = temperature
            data["surface_density"] = surface_density
            data["optical_depth"] = optical_depth
            data["emissivity"] = emissivity
            data["brightness"] = brightness

            if make_plots:
                comp_dir.mkdir(exist_ok=True, parents=True)
                plots = [temperature, surface_density, optical_depth, emissivity, brightness]
                names = ["temperature", "surface_density", "optical_depth",
                         "emissivity", "brightness"]
                for name, plot in zip(names, plots):
                    for wl_index, values in enumerate(plot):
                        plt.plot(radius[0], values, label=wavelength[wl_index])
                    plt.title(f"{component.shortname} {name.title()}")
                    plt.legend()
                    plt.savefig(comp_dir / f"{component.shortname}_{name}.png")
                    plt.close()

        flux = component.compute_flux(wavelength[:, np.newaxis])
        # TODO: Plot this
        data["flux"] = flux
        data["flux_ratio"] = np.zeros(data["flux"].shape)

        total_flux += u.Quantity(data["flux"].squeeze(), unit=u.Jy)

        for parameter in component.get_params().values():
            if parameter.name == "flux":
                continue
            if parameter.wavelength is None:
                name = parameter.shortname.upper()
                if name not in table_header:
                    description = f"[{parameter.unit}] {parameter.description}"
                    table_header[name] = (parameter().value, description)
            else:
                data[parameter.name] = parameter(wavelength[:, np.newaxis])

        table = fits.BinTableHDU(
                Table(data=data),
                name="_".join(component_labels[index].split(" ")).upper(),
                header=table_header)
        tables.append(table)

    data = None
    for table in tables:
        flux_ratio = (table.data["flux"].squeeze() * u.Jy/total_flux)[:, np.newaxis]
        flux_ratio = np.round(flux_ratio * 100, 2)
        table.data["flux_ratio"] = flux_ratio
        if table.header["COMP"] in ["Star", "Point"]:
            continue
        if data is None:
            data = {col.name: table.data[col.name] for col in table.columns}
            continue
        for column in table.columns:
            if column.name in ["wavelength", "kappa_abs", "kappa_cont", "flux"]:
                continue
            if column.name == "radius":
                filler = np.tile(
                        np.linspace(data[column.name][0][-1], table.data[column.name][0][0], dim),
                        (table.data[column.name].shape[0], 1))
            else:
                filler = np.zeros(data[column.name].shape)
            data[column.name] = np.hstack((data[column.name],
                                           filler, table.data[column.name]))

    table = fits.BinTableHDU(Table(data=data), name="FULL_DISK")
    tables.append(table)

    if opacities is not None:
        data = {"wavelength": opacities[np.argmin([op.wavelength.size for op in opacities])].wavelength}
        for opacity in opacities:
            data[opacity.shortname] = np.interp(data["wavelength"], opacity.wavelength, opacity())
        tables.append(fits.BinTableHDU(Table(data=data), name="OPACITIES"))

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = (*np.array(image.shape[1:]) // 2, wavelength.size)
    wcs.wcs.cdelt = (-pixel_size.to(u.deg).value, -pixel_size.to(u.deg).value, -1.)
    wcs.wcs.crval = (0.0, 0.0, 1.0)
    # wcs.wcs.ctype = ("RA---DEG", "DEC--DEG", "WAVELENGTHS")
    wcs.wcs.cunit = ("deg", "deg", "um")
    wcs.wcs.pc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    header = wcs.to_header()

    header["FITMETH"] = (OPTIONS.fit.method, "Method applied for fitting")
    if OPTIONS.fit.method == "emcee":
        header["NBURNIN"] = (nburnin, "Number of burn-in steps for emcee")
        header["NSTEP"] = (nsteps, "Number of steps for emcee")
        header["NWALK"] = (nwalkers, "Numbers of walkers for emcee")
    else:
        header["NLIVE"] = (nlive, "Number of burn-in steps for dynesty")
        header["NSTEP"] = (sample, "Method of sampling for dynesty")
        header["NWALK"] = (bound, "Method of bounding for dynesty")

    header["NCORE"] = (ncores, "Numbers of cores for the fitting")
    header["OBJECT"] = (object_name, "Name of the object")
    header["DATE"] = (f"{datetime.now()}", "Creation date")
    header["R0"] = (OPTIONS.model.reference_radius.value,
                    "[AU] Reference radius for the power laws")

    hdu = fits.HDUList([fits.PrimaryHDU(image, header=header), *tables])
    hdu.writeto(save_dir / "mode.fits", overwrite=True)
