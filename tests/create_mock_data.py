from pathlib import Path

import astropy.units as u

from ppdmod.model import Model
from ppdmod.custom_components import assemble_components
from ppdmod.synthetic import create_synthetic
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS
from ppdmod.utils import get_next_power_of_two


path = Path("data/fits/hd163296/")
fits_file = ["HD_163296_2019-03-23T08_41_19_L_TARGET_FINALCAL_INT.fits"]
fits_file = list(map(lambda x: path / x, fits_file))[0]

fov, pixel_size = 220, 0.1
dim = get_next_power_of_two(fov / pixel_size)

OPTIONS["model.constant_params"] = {
        "dim": dim, "dist": 101, "eff_temp": 9250,
        "pixel_size": pixel_size, "eff_radius": 1.6, "inner_temp": 1500}

rin = Parameter(**STANDARD_PARAMETERS["rin"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])

rin.value = 3.33
a.value = 0.5
phi.value = 130

rin.set(min=0, max=4)
a.set(min=0., max=1.)
phi.set(min=0, max=360)

inner_ring = {"rin": rin, "a": a, "phi": phi}

q = Parameter(**STANDARD_PARAMETERS["q"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])

q.value = 0.5
pa.value = 145
elong.value = 0.5

q.set(min=0, max=1.)
pa.set(min=0, max=360)
elong.set(min=0, max=1)

OPTIONS["model.shared_params"] = {"q": q, "pa": pa, "elong": elong}
OPTIONS["model.components_and_params"] = [
        ["Star", {}],
        ["AsymmetricImageOpticallyThickGradient", inner_ring],
]

OPTIONS["model.matryoshka"] = True
OPTIONS["model.matryoshka.binning_factors"] = [2, 0, 1]

components = assemble_components(
    OPTIONS["model.components_and_params"], OPTIONS["model.shared_params"])
synthetic_path = Path("data/fits/synthetic")
if not synthetic_path.exists():
    synthetic_path.mkdir()

m = Model(components)
create_synthetic(m, fits_file, 0.1*u.mas, synthetic_path)
