import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from ppdmod.components import Ring
from ppdmod.plot import plot_components

if __name__ == "__main__":
    au_to_mas = lambda x: x * 1e3 / 158.51
    ring = Ring(
        label="Ring",
        rin=au_to_mas(1),
        rout=au_to_mas(3),
        pa=352,
        cinc=np.cos(np.deg2rad(46)),
        has_outer_radius=True,
        thin=False
    )
    plot_components([ring], 2048, 0.1, 10.5, save_as_fits=True, savefig="ring.fits")
