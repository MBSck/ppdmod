from pathlib import Path

import numpy as np

from ppdmod.components import Ring
from ppdmod.plot import make_2D_fourier, plot_components


def au_to_mas(x):
    return x * 1e3 / 158.51


def make_ring(
    r: float,
    theta: float,
    rin: float,
    rout: float,
    width: float,
    pa: float,
    i: float,
    rho: float,
    phi: float,
):
    thin, has_outer_radius, asymmetric = True, False, False
    if rout != 0 or width != 0:
        thin = False
    if rout != 0:
        has_outer_radius = True
    if rho != 0 or phi != 0:
        asymmetric = True

    return Ring(
        label="Ring",
        r=au_to_mas(r),
        theta=theta,
        rin=au_to_mas(rin),
        rout=au_to_mas(rout),
        width=au_to_mas(width),
        rho1=rho,
        theta1=phi,
        pa=pa,
        cinc=np.cos(np.deg2rad(i)),
        thin=thin,
        has_outer_radius=has_outer_radius,
        asymmetric=asymmetric,
    )


def save_ring(
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
    image_dir = Path("images")
    ring = make_ring(r, phi, rin, rout, width, pa, i, rho, theta)
    # TODO: Make this more automated
    fits_file = (
        f"ring_r{r}au_phi{phi}deg_rin{rin}au_rout{rout}au"
        f"_width{width}au_pa{pa}deg_i{i}deg_rho{rho}_theta{theta}deg.fits"
    )

    plot_components(
        [ring], 4096, 0.1, 3.58881, save_as_fits=True, savefig=image_dir / fits_file
    )
    # make_2D_fourier([ring], 128, 10.5, savefig=f"fourier/{file_name}")


if __name__ == "__main__":
    save_ring(r=0, phi=0, rin=1, rout=0, width=0, pa=0, i=0, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=0, width=0, pa=352, i=46, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=0, width=0.5, pa=0, i=0, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=0, width=0.5, pa=352, i=46, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=0, i=0, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=0, i=0, rho=1, theta=46)
    save_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=352, i=46, rho=0, theta=0)
    save_ring(r=0, phi=0, rin=1, rout=1.5, width=0, pa=352, i=46, rho=1, theta=46)
