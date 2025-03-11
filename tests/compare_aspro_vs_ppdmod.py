from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.io import fits
from make_image import make_ring

from ppdmod.utils import compute_t3, compute_vis


def read_fits_to_namespace(fits_file: Path) -> SimpleNamespace:
    namespace = SimpleNamespace()
    vis, t3 = SimpleNamespace(), SimpleNamespace()
    with fits.open(fits_file) as hdul:
        namespace.wl = hdul["oi_wavelength"].data["eff_wave"]
        vis.val = hdul["oi_vis"].data["visamp"]
        vis.u = np.round(hdul["oi_vis"].data["ucoord"], 2)
        vis.v = np.round(hdul["oi_vis"].data["vcoord"], 2)
        namespace.vis = vis
        t3.val = hdul["oi_t3"].data["t3phi"]
        u1, u2 = hdul["oi_t3"].data["u1coord"], hdul["oi_t3"].data["u2coord"]
        v1, v2 = hdul["oi_t3"].data["v1coord"], hdul["oi_t3"].data["v2coord"]
        t3.u123 = np.round(np.array([u1, u2, u1 + u2]), 2)
        t3.v123 = np.round(np.array([v1, v2, v1 + v2]), 2)
        t3.u, t3.v = np.unique(t3.u123), np.unique(t3.v123)
        t3.i123 = np.vectorize(lambda x: np.where(t3.u == x)[0][0])(t3.u123)
        namespace.t3 = t3

    return namespace


def compare_ring(
    r: float,
    phi: float,
    rin: float,
    rout: float,
    width: float,
    pa: float,
    i: float,
    rho: float,
    theta: float,
):
    aspro_dir = Path("aspro")
    fits_file = (
        f"ring_r{r}au_phi{phi}deg_rin{rin}au_rout{rout}au"
        f"_width{width}au_pa{pa}deg_i{i}deg_rho{rho}_theta{theta}deg.fits"
    )
    wl, wl_ind = 3.58881 * u.um, 31
    data = read_fits_to_namespace(aspro_dir / fits_file)
    assert np.isclose(wl.value * 1e-6, data.wl[wl_ind])

    ring = make_ring(r, phi, rin, rout, width, pa, i, rho, theta)
    # TODO: Check this calculation
    vis = compute_vis(
        ring.compute_complex_vis(data.vis.u * u.m, data.vis.v * u.m, wl).T
    ).squeeze(-1)
    # TODO: Check this calculation
    t3 = compute_t3(
        np.transpose(
            ring.compute_complex_vis(data.t3.u * u.m, data.t3.v * u.m, wl).T[
                data.t3.i123
            ],
            (1, 0, 2),
        )
    ).squeeze(-1)
    assert np.allclose(data.vis.val[:, wl_ind], vis, atol=1e-1)
    # assert np.allclose(t3.vis.val[:, wl_ind], t3, atol=1e-1)


# TODO: Do the same with some saved components and compare the results
if __name__ == "__main__":
    compare_ring(r=0, phi=0, rin=1, rout=0, width=0, pa=0, i=0, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=0, width=0, pa=352, i=46, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=0, width=0.5, pa=0, i=0, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=0, width=0.5, pa=352, i=46, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=0, i=0, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=0, i=0, rho=1, theta=46)
    # compare_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=352, i=46, rho=0, theta=0)
    # compare_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=352, i=46, rho=1, theta=46)
