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

# TODO: Make function that saves model parameters to load.

# NOTE: Turns off numpys automated parellelization.
os.environ["OMP_NUM_THREADS"] = "1"

# TODO: Check wavelength axis for opacity interpolation.
data.set_fit_wavelengths([3.520375, 10.001093]*u.um)
path = Path("tests/data/fits/")
fits_files = [
    "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits",
    "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits",
    "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT.fits",
    "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
fits_files = list(map(lambda x: path / x, fits_files))
data.set_data(fits_files)

wavelength_axes = list(
    map(lambda x: data.ReadoutFits(x).wavelength, fits_files))
wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
qval_file_dir = Path("tests/data/qval")
qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
opacity = utils.linearly_combine_opacities(
    weights, qval_paths, wavelength_axes)
continuum_opacity = utils.opacity_to_matisse_opacity(
    wavelength_axes, qval_file=qval_file_dir / "Q_amorph_c_rv0.1.dat")

kappa_abs = Parameter(name="kappa_abs", value=opacity,
                      wavelength=wavelength_axes,
                      unit=u.cm**2/u.g, free=False,
                      description="Dust mass absorption coefficient")
kappa_cont = Parameter(name="kappa_cont", value=continuum_opacity,
                       wavelength=wavelength_axes,
                       unit=u.cm**2/u.g, free=False,
                       description="Continuum dust mass absorption coefficient")

fov, pixel_size = 220, 0.1
dim = utils.get_next_power_of_two(fov / pixel_size)

OPTIONS["model.constant_params"] = {
    "dim": dim, "dist": 150, "eff_temp": 7500,
    "pixel_size": pixel_size, "eff_radius": 1.8,
    "inner_temp": 1500, "kappa_abs": kappa_abs, "kappa_cont": kappa_cont}

rin = Parameter(**STANDARD_PARAMETERS["rin"])
rout = Parameter(**STANDARD_PARAMETERS["rout"])

rin.value = 3.33
rout.value = 7.

rin.set(min=3, max=6)
rout.set(min=6, max=10)
rout.free = True

inner_ring = {"rin": rin, "rout": rout}
inner_ring_labels = [f"ir_{label}" for label in inner_ring]

rin = Parameter(**STANDARD_PARAMETERS["rin"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])

rin.value = 13
a.value = 0.5
phi.value = 130

rin.set(min=7, max=30)
a.set(min=0., max=1.)
phi.set(min=0, max=360)

outer_ring = {"rin": rin, "a": a, "phi": phi}
outer_ring_labels = [f"or_{label}" for label in outer_ring]

p = Parameter(**STANDARD_PARAMETERS["p"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])
cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

p.value = 0.5
pa.value = 145
elong.value = 0.5
cont_weight.value = 130
inner_sigma.value = 1e-3

p.set(min=0., max=1.)
pa.set(min=0, max=360)
elong.set(min=0, max=1)
cont_weight.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)

OPTIONS["model.shared_params"] = {"p": p, "pa": pa, "elong": elong,
                                  "inner_sigma": inner_sigma,
                                  "cont_weight": cont_weight}
shared_params_labels = [f"sh_{label}"
                        for label in OPTIONS["model.shared_params"]]

OPTIONS["model.components_and_params"] = [
    ["Star", {}],
    ["SymmetricSDGreyBodyContinuum", inner_ring],
    ["AsymmetricSDGreyBodyContinuum", outer_ring],
]

# OPTIONS["fourier.binning"] = 3
# OPTIONS["fourier.padding"] = 3
# print("Binned Dimension",
#       dim*2**-OPTIONS["fourier.binning"],
#       "Resolution",
#       pixel_size*2**OPTIONS["fourier.binning"])
# print("Binned and padded Dimension",
#       dim*2**-OPTIONS["fourier.binning"]*2**OPTIONS["fourier.padding"])

OPTIONS["model.matryoshka"] = True
OPTIONS["model.matryoshka.binning_factors"] = [2, 0, 1]

labels = inner_ring_labels + outer_ring_labels + shared_params_labels

components = custom_components.assemble_components(
    OPTIONS["model.components_and_params"], OPTIONS["model.shared_params"])
m = model.Model(components)
image = m.calculate_image(4096, 0.1, OPTIONS["fit.wavelengths"][-1])
fits_filename = 'test_model.fits'
hdu = fits.PrimaryHDU(image.value)
hdu.writeto(fits_filename, overwrite=True)
plot.plot_model(4096, 0.1, m, OPTIONS["fit.wavelengths"][-1], zoom=None)
breakpoint()

OPTIONS["fourier.binning"] = 3


if __name__ == "__main__":
    nburnin, nsteps, nwalkers_burnin, nwalkers = 100, 2500, 200, 35

    model_result_dir = Path("/Users/scheuck/Data/model_results/")
    day_dir = model_result_dir / str(datetime.now().date())
    time = datetime.now()
    file_name = f"results_model_nsteps{nburnin+nsteps}_nwalkers{nwalkers}"\
            f"_{time.hour}:{time.minute}:{time.second}"
    result_dir = day_dir / file_name
    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    sampler = mcmc.run_mcmc(nwalkers, nwalkers_burnin, nsteps, nburnin)
    theta = mcmc.get_best_fit(sampler, discard=nburnin)
    np.save(result_dir / "best_fit_params.npy", theta)
    new_params = dict(zip(labels, theta))

    plot.plot_chains(sampler, labels, discard=nburnin, savefig=result_dir / "chains.pdf")
    plot.plot_corner(sampler, labels, discard=nburnin, savefig=result_dir / "corner.pdf")
    # OPTIONS["fourier.binning"] = None
    # OPTIONS["fourier.padding"] = None
    OPTIONS["model.matryoshka"] = False
    wavelength = OPTIONS["fit.wavelengths"][1]
    components_and_params, shared_params = mcmc.set_params_from_theta(theta)
    components = custom_components.assemble_components(
        components_and_params, shared_params)

    # HACK: This is to include innermost radius for rn.
    innermost_radius = components[1].params["rin"]
    for component in components:
        component.params["rin0"] = innermost_radius

    m = model.Model(components)
    plot.plot_model(4096, 0.1, m, wavelength, zoom=None,
                    savefig=None)
    plot.plot_observed_vs_model(m, 0.1*u.mas, new_params["sh_elong"],
                                new_params["sh_pa"],
                                savefig=result_dir / "fit_results.pdf")
