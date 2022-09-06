import os
import numpy as np
import astropy.units as u

from astropy.io import fits
from pathlib import Path
from astropy.units import Quantity
from typing import Any, Tuple, Dict, List, Union
from scipy.interpolate import CubicSpline

# TODO: Make merge function that merges different readouts
# TOOD: Make Readout accept more than one file


def read_single_dish_txt2np(file, axis2interpolate):
    """Reads x, y '.txt'-file intwo 2 numpy arrays"""
    file_data = np.loadtxt(file)
    wavelength_axis = np.array([i[0] for i in file_data])*1e-6
    flux_axis = np.array([i[1] for i in file_data])

    wl2flux_dict = {}
    cubic_spline = CubicSpline(wavelength_axis, flux_axis)
    for i, wl_axis in enumerate(cubic_spline(axis2interpolate)):
        wl2flux_dict[axis2interpolate[i]] = wl_axis

    return wl2flux_dict

class ReadoutFits:
    """All functionality to work with (.fits)-files"""
    def __init__(self, fits_files: List[Path]) -> None:
        self.fits_file = fits_files

    def get_info(self) -> str:
        """Gets the (.fits)-file's primary header's info

        Returns
        -------
        primary_header_content: str
        """
        with fits.open(self.fits_file) as header_list:
            return header_list.info()

    def get_header(self, header: Union[int, str]) -> str:
        """Reads out the data of the header

        Parameters
        ----------
        header: int | str
            The header of the data to be retrieved

        Returns
        -------
        header_content: str
        """
        return repr(fits.getheader(self.fits_file, header))

    def get_data(self, header: Union[int, str],
                 *sub_headers: Union[int, str]) -> List[np.array]:
        """Gets a specific set of data and its error from a header and
        subheader and returns the data of as many subheaders as in args

        Parameters
        ----------
        header: int | str
            The header of the data to be retrieved
        sub_headers: int | str
            The subheader(s) that specify the data

        Returns
        -------
        data: List[np.array]
        """
        with fits.open(self.fits_file) as header_list:
            return [header_list[header].data[sub_header] for sub_header in sub_headers]

    def get_column_names(self, header: Union[int, str]) -> np.ndarray:
        """Fetches the columns of the header

        Parameters
        ----------
        header: int | str
            The header of the data to be retrieved

        Returns
        -------
        column_names: List[np.array]
        """
        with fits.open(self.fits_file) as header_list:
            return (header_list[header].columns).names

    def get_uvcoords(self) -> Quantity:
        """Fetches the u, v coordinates from the (.fits)-files, merges them and gives the
        quantities the proper units

        Returns
        -------
        uvcoords: astropy.units.Quantity
            The (u, v)-coordinates in [astropy.units.m]
        """
        ucoords = np.array(self.get_data("oi_vis", "ucoord"))
        vcoords = np.array(self.get_data("oi_vis", "vcoord"))
        return np.array([uvcoords for uvcoords in zip(ucoords, vcoords)])*u.m

    def get_split_uvcoords(self) -> Tuple[Quantity]:
        """Fetches the u, v coordinates from the (.fits)-files and gives the
        quantities the proper units

        Returns
        -------
        ucoords: astropy.units.Quantity
            The u-coordinates in [astropy.units.m]
        vcoords: astropy.untis.Quantity
            The v-coordinates in [astropy.units.m]
        """
        ucoords = np.array(self.get_data("oi_vis", "ucoord"))*u.m
        vcoords = np.array(self.get_data("oi_vis", "vcoord"))*u.m
        return ucoords, vcoords

    def get_closure_phase_uvcoords(self) -> Tuple[Quantity]:
        """Fetches the (u1, v1), (u2, v2) coordinate of the closure phase triangles from
        the (.fits)-file, calculates the third (u3, v3) coordinate pair and then gives the
        quantities the proper units

        Returns
        -------
        u: Tuple[astropy.units.Quantity]
            The three u-coordinate pairs of the closure phase triangles
        v: Tuple[astropy.units.Quantity]
            The three v-coordinate pairs of the closure phase triangles
        """
        u1, v1 = self.get_data("oi_t3", "u1coord", "v1coord")*u.m
        u2, v2 = self.get_data("oi_t3", "u2coord", "v2coord")*u.m
        u3, v3 = -(u1+u2), -(v1+v2)
        return [u1, u2, u3], [v1, v2, v3]

    def get_baselines(self) -> Quantity:
        """Calculates the baselines from the uv coordinates

        Returns
        -------
        baselines: astropy.unit.Quantity
            The baselines in [astropy.units.meter]
        """
        return np.sqrt(*self.get_split_uvcoords**2)

    def get_visibilities(self) -> Quantity:
        """"Fetches the visibility data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities.

        Returns
        -------
        vis: astropy.units.Quantity
            The visibility of an observed object either in [astropy.units.Jansky] or
            [astropy.units.dimensionless_unscaled]
        viserr: astropy.units.Quantity
            The error of the visibility of an observed object either in
            [astropy.units.Jansky]
            or [astropy.units.dimensionless_unscaled]
        sta_indicies: astropy.units.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        vis, viserr, sta_indicies = self.get_data("oi_vis", "visamp",\
                "visamperr", "sta_index")

        if np.max(vis) > 1.:
            vis, viserr = map(lambda x: x*u.Jy, [vis, viserr])
        else:
            vis, viserr = map(lambda x: x*u.dimensionless_unscaled, [vis, viserr])
        sta_indicies *= u.dimensionless_unscaled
        return vis, viserr, sta_indicies

    def get_visibilities_squared(self) -> Quantity:
        """Fetches the squared visibility data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities

        Returns
        ----------
        vis2: astropy.units.Quantity
            The squared visibility of an observed object in
            [astropy.units.dimensionless_unscaled]
        vis2err: astropy.units.Quantity
            The error of the squared visibility of an observed object in
            [astropy.units.dimensionless_unscaled]
        sta_indicies: astropy.units.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        vis2, vis2err, sta_indicies = self.get_data("oi_vis2", "vis2data",
                "vis2err", "sta_index")
        return map(lambda x: x*u.dimensionless_unscaled, [vis2, vis2err, sta_indicies])

    def get_closure_phases(self) -> Quantity:
        """Fetches the closure phase data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities

        Returns
        ----------
        cphases: u.Quantity
            The closure phases of an observed object in [astropy.units.degree]
        cphaseserr: u.Quantity
            The error of the closure phases of an observed object in
            [astropy.units.degree]
        sta_indicies: u.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        cphases, cphaseserr, sta_indicies = self.get_data("oi_t3", "t3phi",
                "t3phierr", "sta_index")
        cphases, cphaseserr = map(lambda x: x*u.deg, [cphases, cphaseserr])
        sta_indicies *= u.dimensionless_unscaled
        return cphases, cphaseserr, sta_indicies

    def get_flux(self) -> Quantity:
        """Fetches the (total) flux data, its error from the (.fits)-file and gives the
        proper units to the quantities

        Returns
        ----------
        flux: u.Quantity
            The (total) flux of an observed object [astropy.units.Jansky]
        fluxerr: u.Quantity
            The error of the (total) flux of an observed object in [astropy.units.Jansky]
        """
        return map(lambda x: x*u.Jy, self.get_data("oi_flux", "fluxdata", "fluxerr"))

    def get_wavelength_solution(self) -> Quantity:
        """Fetches the wavelength solution from the (.fits)-file and gives the
        proper units to the quantities

        Returns
        ----------
        flux: astropy.units.Quantity
            The wavelength solution of the MATISSE instrument [astropy.units.micrometer]
        """
        return self.get_data("oi_wavelength", "eff_wave")[0]

    def get_tel_sta(self) -> np.ndarray:
        """Fetches the telescope array names and station from the (.fits)-files and
        gives the proper units to the quantities

        Returns
        -------
        station_name: astropy.units.Quantity
        station_indicies: astropy.units.Quantity
        """
        return self.get_data("oi_array", "tel_name", "sta_index")

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

