#!/usr/bin/env python3

import os
import numpy as np

from scipy.interpolate import CubicSpline
from astropy.io import fits
from typing import Any, Dict, List, Union, Optional

def read_single_dish_txt2np(file, wl_axis):
    """Reads x, y '.txt'-file intwo 2 numpy arrays"""
    file_data = np.loadtxt(file)
    wavelength_axis = np.array([i[0] for i in file_data])*1e-6
    flux_axis = np.array([i[1] for i in file_data])

    wl2flux_dict = {}
    cs = CubicSpline(wavelength_axis, flux_axis)
    for i, o in enumerate(cs(wl_axis)):
        wl2flux_dict[wl_axis[i]] = o

    return wl2flux_dict

class ReadoutFits:
    """All functionality to work with '.oifits/.fits'-files"""
    def __init__(self, fits_file) -> None:
        self.fits_file = fits_file

    def get_info(self) -> str:
        """Gets the header's info"""
        with fits.open(self.fits_file) as hdul:
            return hdul.info()

    def get_header(self, hdr) -> str:
        """Reads out the specified data"""
        return repr(fits.getheader(self.fits_file, hdr))

    def get_data(self, hdr: Union[int, str], *args: Union[int, str]) -> List[np.array]:
        """Gets a specific set of data and its error from a header and
        subheader and returns the data of as many subheaders as in args

        Parameters
        ----------
        hdr: int | str
            The header of the data to be retrieved
        args: int | str
            The subheader(s) that specify the data

        Returns
        -------
        data: List[np.array]
        """
        with fits.open(self.fits_file) as hdul:
            return [hdul[hdr].data[i] for i in args] if len(args) > 1 \
                    else hdul[hdr].data[args[0]]

    def get_column_names(self, hdr) -> np.ndarray:
        """Fetches the columns of the header"""
        with fits.open(self.fits_file) as hdul:
            return (hdul[hdr].columns).names

    def get_uvcoords(self) -> np.ndarray:
        """Fetches the u, v coord-lists and merges them as well as the individual components"""
        return [i for i in zip(self.get_data(4, "ucoord")[:6], self.get_data(4, "vcoord")[:6])]

    def get_split_uvcoords(self) -> np.ndarray:
        """Splits a 2D-np.array into its 1D-components and returns the u- and
        v-coords seperatly"""
        uvcoords = self.get_uvcoords()
        return [item[0] for item in uvcoords], [item[1] for item in uvcoords]

    def get_t3phi_uvcoords(self):
        """Fetches the (u1, v1), (u2, v2) coordinate tuples and then calculates
        the corresponding baselines

        Returns
        -------
        u: np.ndarray
        v: np.ndarray
        """
        u1, v1 = self.get_data("oi_t3", "u1coord", "v1coord")
        u2, v2 = self.get_data("oi_t3", "u2coord", "v2coord")
        u3, v3 = -(u1+u2), -(v1+v2)
        return [u1, u2, u3], [v1, v2, v3]

    def get_baselines(self):
        """Calculates the baselines from the uv coordinates"""
        u, v = self.get_split_uvcoords()
        return np.sqrt(u**2+v**2)

    def get_vis(self) -> np.ndarray:
        """Fetches the visibility data/correlated fluxes, its errors and sta-indices"""
        return self.get_data("oi_vis", "visamp", "visamperr", "sta_index")

    def get_vis2(self) -> np.ndarray:
        """Fetches the squared visibility data, its error and sta_indicies"""
        return self.get_data("oi_vis2", "vis2data", "vis2err", "sta_index")

    def get_t3phi(self) -> np.ndarray:
        """Fetches the closure phase data, its error and sta_indicies"""
        return self.get_data("oi_t3", "t3phi", "t3phierr", "sta_index")

    def get_flux(self) -> np.ndarray:
        """Fetches the flux"""
        return map(lambda x: x[0], self.get_data("oi_flux", "fluxdata",
                                                 "fluxerr"))
    def get_wl(self) -> np.ndarray:
        return self.get_data("oi_wavelength", "eff_wave")

    def get_tel_sta(self) -> np.ndarray:
        return self.get_data(2, "tel_name", "sta_index")

    def get_flux4wl(self, wl_ind: np.ndarray) -> np.ndarray:
        """Fetches the flux for a specific wavelength"""
        return map(lambda x: x[wl_ind], self.get_flux())

    def get_vis4wl(self, wl_ind: np.ndarray) -> np.ndarray:
        """Fetches the visdata(amp/phase)/correlated fluxes for one specific wavelength

        Parameters
        ----------
        wl_ind: np.ndarray

        Returns
        --------
        visamp4wl: np.ndarray
            The visamp for a specific wavelength
        visamperr4wl: np.ndarray
            The visamperr for a specific wavelength
        """
        visdata = self.get_vis()
        visamp, visamperr = map(lambda x: x[:6], visdata[:2])

        # FIXME: Is this ordering done correctly?? Check!
        visamp4wl, visamperr4wl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten().tolist(), [visamp, visamperr])

        return visamp4wl, visamperr4wl

    def get_t3phi4wl(self, wl_ind: int) -> np.ndarray:
        """Fetches the closure phases for one specific wavelength

        Returns
        -------
        t3phi4wl: np.ndarray
            The closure phase for a specific wavelength
        t3phierr4wl: np.ndarray
            The closure phase error for a specific wavelength
        """
        t3phi, t3phierr = self.get_t3phi()[:2]
        t3phi4wl, t3phierr4wl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [t3phi, t3phierr])

        return t3phi4wl, t3phierr4wl

    def get_vis24wl(self, wl_ind: np.ndarray) -> np.ndarray:
        """Fetches the vis2data for one specific wavelength

        Returns
        --------
        vis2data4wl: np.ndarray
            The vis2data for a specific wavelength
        vis2err4wl: np.ndarray
            The vis2err for a specific wavelength
        """
        vis2data, vis2err  = map(lambda x: x[:6], self.get_vis2()[:2])
        vis2data4wl, vis2err4wl = map(lambda x: np.array([i[wl_ind] for i in x]).flatten(), [vis2data, vis2err])

        return vis2data4wl, vis2err4wl


if __name__ == "__main__":
    path = "/Users/scheuck/Documents/PhD/matisse_stuff/ppdmodler/assets/data"
    path = os.path.join(path,
                        "HD_142666_2019-03-24T09_01_46_L_TARGET_FINALCAL_INT.fits")
    readout = ReadoutFits(path)
    print(readout.get_vis4wl(30))

