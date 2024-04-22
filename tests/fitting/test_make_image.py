from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from ppdmod import data
from ppdmod.basic_components import assemble_components
from ppdmod.fitting import compute_observable_chi_sq, compute_observables
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import compute_photometric_slope


DATA_DIR = Path("../data/pionier/HD142527")
OPTIONS.model.output = "non-physical"
fits_files = list((DATA_DIR).glob("*fits"))
data.set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 2.66*u.rad.to(u.deg)
pa.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.69
inc.free = True

fs = Parameter(**STANDARD_PARAMETERS.fr)
fs.value = 0.55
fs.free = True

fc = Parameter(**STANDARD_PARAMETERS.fr)
fc.value = 0.34
fc.free = True

wavelength = data.get_all_wavelengths()
ks = Parameter(**STANDARD_PARAMETERS.exp)
ks.value = compute_photometric_slope(wavelength, 7500*u.K)
ks.wavelength = wavelength
ks.free = False

kc = Parameter(**STANDARD_PARAMETERS.exp)
kc.value = -2.43
kc.set(min=-10, max=10)

fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
fwhm.value = 5.2605359837907635
fwhm.set(min=0.1, max=32)

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 2.25288818328784
rin.set(min=0.1, max=32)

flor = Parameter(**STANDARD_PARAMETERS.fr)
flor.value = 1.
flor.free = True

a = Parameter(**STANDARD_PARAMETERS.a)
a.value = 0.2404163056034262 

phi = Parameter(**STANDARD_PARAMETERS.phi)
phi.value = -45.

params = {"fs": fs, "fc": fc, "flor": flor, "fwhm": fwhm,
          "rin": rin, "kc": kc, "inc": inc, "pa": pa, "a": a, "phi": phi}
labels = [label for label in params]

OPTIONS.model.constant_params = {"wl0": 1.68, "ks": ks}
OPTIONS.model.components_and_params = [["StarHaloRing", params]]
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

result_dir = Path("results/pionier")
result_dir.mkdir(exist_ok=True, parents=True)
model_name = "starHaloGaussLorRing"

components = assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)

rchi_sq = compute_observable_chi_sq(
        *compute_observables(components), reduced=True)
print(f"rchi_sq: {rchi_sq}")

image = components[0].compute_image(512, 0.02, 1.68)
plt.imshow(image[0], origin="lower")
plt.show()
