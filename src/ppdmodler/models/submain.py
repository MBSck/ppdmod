import sys
import numpy as np
import matplotlib.pyplot as plt

from src.functionality.readout import ReadoutFits
from src.functionality.fourier import FFT
from src.functionality.utilities import trunc, azimuthal_modulation
from src.models import Gauss2D, Ring, CompoundModel, InclinedDisk, UniformDisk

# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def check_flux_behaviour(model, wavelength):
    """Plots the scaling behaviour of the fluxes"""
    lst = []
    pot_lst = [2**i for i in range(1, 10, 1)][3:]
    print(pot_lst)
    for j in pot_lst:
        mod = model.eval_model([1], 10, j)
        flux = model.get_flux(0.5, 0.55, 1500, 19, 140, wavelength)
        lst.append([j, flux])

    for i, o in enumerate(lst):
        if i == len(lst)//2:
            break

        print("|| ", o[0], ": ", trunc(o[1], 3),
              " || ", lst[~i][0], ": ", trunc(lst[~i][1], 3), " ||")

def check_interpolation(uvcoords, uvcoords_cphase, wavelength):
    u = CompoundModel(1500, 7900, 19, 140, wavelength)
    u_mod = u.eval_model([0.5, 135, 0.5, 180, 1.7, 0.04, 0.6], 20, 1024)

    # Check previously calculated scaling factor
    fft = FFT(u_mod, wavelength, u.pixel_scale, 2)
    amp, cphase, xy_coords = fft.get_uv2fft2(uvcoords, uvcoords_cphase,
                                             intp=True, corr_flux=False)
    fft.plot_amp_phase(corr_flux=False, zoom=1000,
                       uvcoords_lst=xy_coords, plt_save=False)

def main():
    wavelength = 3.5e-6
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/assets/GTO/hd142666/UTs/nband/TAR-CAL.mat_cal_estimates.2019-05-14T05_28_03.AQUARIUS.2019-05-14T04_52_11.rb/averaged/Final_CAL.fits"
    readout = ReadoutFits(path)

    uv = readout.get_uvcoords()
    uv_cphase = readout.get_t3phi_uvcoords()
    check_interpolation(uv, uv_cphase, wavelength)

if __name__ == "__main__":
    main()

