from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1

from data import ReadoutFits
from fft import compute_2Dfourier_transform, interpolate_for_coordinates


def uniform_disk(diameter, pixel_size, dim):
    v = np.linspace(-0.5, 0.5, dim, endpoint=False)*pixel_size*dim
    x_arr, y_arr = np.meshgrid(v, v)
    return 4*(np.hypot(x_arr, y_arr) < diameter/2)/(np.pi*diameter**2)


def uniform_disk_vis(diameter, ucoord, vcoord, wl):
    rho = np.hypot(ucoord, vcoord)/wl
    diameter = diameter*u.mas.to(u.rad)
    return 2*j1(np.pi*rho*diameter)/(np.pi*diameter*rho)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    files = ["hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits",
             "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
    files = list(map(lambda x: path / x, files))
    data = [ReadoutFits(file) for file in files]
    diameter, wavelength = 20, 1.02322101e-05
    dim, pixel_size = 4096, 0.1
    pixel_size_rad = pixel_size*u.mas.to(u.rad)
    model = uniform_disk(diameter, pixel_size, dim)
    ucoord = np.linspace(1287, 1288, 10)/10
    vcoord = ucoord*0
    print(ucoord)
    vis = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    ft = compute_2Dfourier_transform(model)
    interpolated_values = interpolate_for_coordinates(ft, dim, pixel_size_rad,
                                                      ucoord,
                                                      vcoord,
                                                      wavelength)
    print(vis)
    print(interpolated_values/ft[dim//2, dim//2])
