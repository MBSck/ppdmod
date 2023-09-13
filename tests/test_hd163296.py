import os
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from ppdmod import custom_components
from ppdmod import data
from ppdmod import mcmc
from ppdmod import model
from ppdmod import plot
from ppdmod import utils
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


# NOTE: Turns off numpys automated parellelization.
os.environ["OMP_NUM_THREADS"] = "1"
data.set_fit_wavelengths([3.2103345, 3.520375, 3.7025948]*u.um)
path = Path("tests/data/fits/hd163296/")
fits_files = ["HD_163296_2019-03-23T08_41_19_L_TARGET_FINALCAL_INT.fits",
              "HD_163296_2019-05-06T08_19_51_L_TARGET_FINALCAL_INT.fits"]
fits_files = list(map(lambda x: path / x, fits_files))
data.set_data(fits_files)

wavelength_axes = list(
        map(lambda x: data.ReadoutFits(x).wavelength, fits_files))
wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))
breakpoint()

fov, pixel_size = 220, 0.1
dim = utils.get_next_power_of_two(fov / pixel_size)

OPTIONS["model.constant_params"] = {
        "dim": dim, "dist": 101, "eff_temp": 7500,
        "pixel_size": pixel_size, "eff_radius": 1.8, "inner_temp": 1500}

rin = Parameter(**STANDARD_PARAMETERS["rin"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])

rin.value = 3.33
a.value = 0.5
phi.value = 130

rin.set(min=3, max=6)
a.set(min=0., max=1.)
phi.set(min=0, max=360)

inner_ring = {"rin": rin, "a": a, "phi": phi}
inner_ring_labels = [f"ir_{label}" for label in inner_ring]

pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])

pa.value = 145
elong.value = 0.5

pa.set(min=0, max=360)
elong.set(min=0, max=1)

OPTIONS["model.shared_params"] = {"pa": pa, "elong": elong}
shared_params_labels = [f"sh_{label}"
                        for label in OPTIONS["model.shared_params"]]

OPTIONS["model.components_and_params"] = [
        ["Star", {}],
        ["AsymmetricSDGreyBodyContinuum", inner_ring],
]

OPTIONS["model.matryoshka"] = True
OPTIONS["model.matryoshka.binning_factors"] = [2, 0, 1]

labels = inner_ring_labels + shared_params_labels

components = custom_components.assemble_components(
        OPTIONS["model.components_and_params"], OPTIONS["model.shared_params"])
m = model.Model(components)
image = m.calculate_image(4096, 0.1, OPTIONS["fit.wavelengths"][-1])
fits_filename = 'test_model.fits'
hdu = fits.PrimaryHDU(image.value)
hdu.writeto(fits_filename, overwrite=True)
plot.plot_model(4096, 0.1, m, OPTIONS["fit.wavelengths"][-1], zoom=None)

OPTIONS["fourier.binning"] = 3


if __name__ == "__main__":
    nburnin, nsteps, nwalkers_burnin, nwalkers = 100, 2500, 200, 35

    model_result_dir = Path("/Users/scheuck/Data/model_results/")
    day_dir = model_result_dir / str(datetime.now().date())
    result_dir = day_dir / f"results_model_nsteps{nburnin+nsteps}_nwalkers{nwalkers}"
    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    sampler = mcmc.run_mcmc(nwalkers, nwalkers_burnin, nsteps, nburnin)
    theta = mcmc.get_best_fit(sampler, discard=nburnin)
    np.save(result_dir / "best_fit_params.npy", theta)
    new_params = dict(zip(labels, theta))

    plot.plot_chains(sampler, labels, discard=nburnin, savefig=result_dir / "chains.pdf")
    plot.plot_corner(sampler, labels, discard=nburnin, savefig=result_dir / "corner.pdf")
    OPTIONS["model.matryoshka"] = False

    wavelength = OPTIONS["fit.wavelengths"][1]
    components_and_params, shared_params = mcmc.set_params_from_theta(theta)
    components = custom_components.assemble_components(
            components_and_params, shared_params)

    m = model.Model(components)
    plot.plot_model(4096, 0.1, m, wavelength, zoom=None,
                    savefig=None)
    plot.plot_observed_vs_model(m, 0.1*u.mas, new_params["sh_elong"],
                                new_params["sh_pa"],
                                savefig=result_dir / "fit_results.pdf")
