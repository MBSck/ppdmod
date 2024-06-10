from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.basic_components import assemble_components
from ppdmod.fitting import compute_observables, compute_observable_chi_sq
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import compute_photometric_slope, compute_t3


DATA_DIR = Path("../data/pionier/HD142527")
OPTIONS.model.output = "non-physical"
fits_files = list((DATA_DIR).glob("*fits"))
data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 1.20*u.rad.to(u.deg)
pa.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.63
inc.free = True

fs = Parameter(**STANDARD_PARAMETERS.fr)
fs.value = 0.42
fs.free = True

fc = Parameter(**STANDARD_PARAMETERS.fr)
fc.value = 0.55
fc.free = True

wavelength = get_all_wavelengths()
ks = Parameter(**STANDARD_PARAMETERS.exp)
ks.value = compute_photometric_slope(wavelength, 7500*u.K)
ks.wavelength = wavelength
ks.free = False

kc = Parameter(**STANDARD_PARAMETERS.exp)
kc.value = -4.12
kc.set(min=-10, max=10)

fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
fwhm.value = 2*4.5993378575405135
fwhm.set(min=0.1, max=32)

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 8.369419048403874
rin.set(min=0.1, max=32)

flor = Parameter(**STANDARD_PARAMETERS.fr)
flor.value = 1.
flor.free = True

a = Parameter(**STANDARD_PARAMETERS.a)
a.value = 0.996393496566492

phi = Parameter(**STANDARD_PARAMETERS.phi)
phi.value = 100.40771131249006

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

image = components[0].compute_image(2048, 0.02, 1.68)
t3 = components[0].compute_complex_vis(data.t3.u123coord, data.t3.v123coord, [1.68]*u.um)
t3 = compute_t3(t3)
print(t3.max())

plt.imshow(image[0])
plt.show()
