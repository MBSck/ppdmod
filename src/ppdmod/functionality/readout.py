import os
import numpy as np
import astropy.units as u

from astropy.io import fits
from pathlib import Path
from astropy.units import Quantity
from typing import Tuple, Dict, List, Optional, Union, Callable
from scipy.interpolate import CubicSpline

# TODO: Make merge function that merges different readouts
# TOOD: Make Readout accept more than one file


# TODO: Make get_band_information method to check the band
class ReadoutFits:
    """All functionality to work with (.fits)-files"""
    def __init__(self, fits_file: Path,
                 flux_file: Optional[Path] = None) -> None:
        self.fits_file = fits_file
        self.flux_file = flux_file
        self.wavelength_solution = self.get_wavelength_solution()

    def __str__(self):
        return f"Readout initialised with (.fits)-file:\n{self.fits_file}"

    def __repr__(self):
        return f"Readout initialised with (.fits)-file:\n{self.fits_file}"

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

    def get_column_names(self, header: Union[int, str]) -> np.ndarray:
        """Fetches the columns of the header

        Parameters
        ----------
        header: int | str
            The header of the data to be retrieved

        Returns
        -------
        column_names: numpy.ndarray
        """
        with fits.open(self.fits_file) as header_list:
            return (header_list[header].columns).names

    def _get_flux_file_data(self) -> Quantity:
        """Reads the flux data from the flux file and then interpolates it to the
        wavelength solution used by MATISSE.

        Returns
        -------
        flux: astropy.units.Quantity
            The flux provided by the flux_file interpolated to MATISSE's wavelength
            solution
        """
        # TODO: Get a better error representation for the flux
        single_dish_data = np.loadtxt(self.flux_file)
        wavelength_from_single_dish = [wl[0] for wl in single_dish_data]
        flux_from_single_dish = [flux[1] for flux in single_dish_data]
        mean_wl = np.mean(wavelength_from_single_dish)

        if all([i for i in self.get_wavelength_indices([mean_wl])]):
            cubic_spline = CubicSpline(wavelength_from_single_dish*u.um,
                                       flux_from_single_dish*u.Jy)
            flux = cubic_spline(self.wavelength_solution)*u.Jy
            return [flux, flux*0.1]
        else:
            raise IOError("The flux file seems to be outside of the wavelength solutions"\
                          " range!")

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
        data: List[numpy.ndarray]
        """
        with fits.open(self.fits_file) as header_list:
            return [header_list[header].data[sub_header] for sub_header in sub_headers]

    def get_wavelength_indices(self, selected_wavelengths: List[float],
                               wavelength_window_sizes:\
                               List[float] = [0.2]) -> List[List[float]]:
        """Fetches the wavelength indices of the instrument's wavelength solution for a
        specific wavelength by taking a window around the chosen wavelength. BEWARE: The
        window is divided by 2 and that is taken in both directions

        Parameters
        ----------
        selected_wavelengths: List[float]
            The wavelengths to be polychromatically fitted. Input will be converted to
            [astropy.units.micrometer]
        wavelength_window_sizes: List[float]
            This determines how far around the central chosen wavelength other
            wavelengths are to be fetched. Input will be converted to
            [astropy.units.micrometer]

        Returns
        -------
        wavelength_indices: List[List[float]]
            A numpy array of wavelength indices for the input wavelengths around the
            window
        """
        if (not len(selected_wavelengths) == 1) and (len(wavelength_window_sizes) == 1):
            # NOTE: If done via normal list multiplication -> Error. Why?! Dunno...
            wavelength_window_sizes = (np.ones((len(selected_wavelengths)))\
                                       *wavelength_window_sizes[0]).tolist()

        if len(wavelength_window_sizes) != len(selected_wavelengths):
            raise IOError("The specified wavelength windows have be the same length"\
                          " as the selected wavelength list!")

        selected_wavelengths *= u.um
        wavelength_window_sizes *= u.um

        window_top_bound = selected_wavelengths + wavelength_window_sizes/2
        window_bot_bound = selected_wavelengths - wavelength_window_sizes/2
        windows = [(self.wavelength_solution > bot, self.wavelength_solution < top)\
                   for bot, top in zip(window_bot_bound, window_top_bound)]

        return [np.where(np.logical_and(*window))[0].tolist() for window in windows]

    def get_data_for_wavelength(self, data: Union[Quantity, np.ndarray],
                                wl_poly_indices: List) -> List:
        """Fetches data for one or more wavelengths from the nested arrays. Gets the
        corresponding values by index from the nested arrays (baselines/triangle)

        Parameters
        ----------
        data: astropy.units.Quantity | numpy.ndarray
            The data for every baseline/triangle
        wl_poly_indices: List
            The polychromatic indices of the wavelength solution. This has to be a doubly
            nested list

        Returns
        --------
        data4wl: List
        """
        # NOTE: Right now the data is immediately averaged after getting taken. Maybe
        # change this for the future
        polychromatic_data_averaged = []
        for dataset in data:
            data4wl = []
            for wl_indices in wl_poly_indices:
                data4wl_poly_index = []
                for wl_index in wl_indices:
                    array_wl_slice = u.Quantity([array[wl_index] for array in dataset])
                    data4wl_poly_index.append(array_wl_slice)
                data4wl.append(u.Quantity(data4wl_poly_index))
            averaged_dataset_slice = self.average_polychromatic_data(data4wl)
            polychromatic_data_averaged.append(averaged_dataset_slice)
        return [u.Quantity(dataset4wl) for dataset4wl in polychromatic_data_averaged]

    def average_polychromatic_data(self, polychromatic_data: Quantity):
        """Fetches and then averages over polychromatic data. Iterates over the
        polychromatic wavelength slices and then takes the mean of them

        Parameters
        ----------
        polychromatic_data: astropy.units.Quantity
            The polychromatic data slices of wavelengths in one window
        """
        return u.Quantity([np.mean(data_slice, axis=0)\
                           for data_slice in polychromatic_data])

    # TODO: Write test for this
    def get_telescope_information(self) -> Union[np.ndarray, Quantity]:
        """Fetches the telescop's array names and stations from the (.fits)-files and
        gives the proper units to the quantities

        Returns
        -------
        station_name: numpy.ndarray
            The names of the four telescopes used
        station_indices: astropy.units.Quantity
            The station indices of the four telescopes used
            [astropy.units.dimensionless_unscaled]
        station_indices4baselines: astropy.units.Quantity
            The station indices of the baselines
            [astropy.units.dimensionless_unscaled]
        station_indices4triangles: astropy.units.Quantity
            The station indices of the closure phases' triangles
            [astropy.units.dimensionless_unscaled]
        """
        station_names, station_indices = self.get_data("oi_array",
                                                        "tel_name", "sta_index")
        station_indices *= u.dimensionless_unscaled
        station_indices4baselines = self.get_data("oi_vis", "sta_index")[0]*\
            u.dimensionless_unscaled
        station_indices4triangles = self.get_data("oi_t3", "sta_index")[0]*\
            u.dimensionless_unscaled

        return np.array([station_names, station_indices,\
            station_indices4baselines, station_indices4triangles], dtype=object)

    # TODO: Write test for this
    def get_split_uvcoords(self) -> Tuple[Quantity]:
        """Fetches the u, v coordinates from the (.fits)-files and gives the
        quantities the proper units

        Returns
        -------
        ucoords: astropy.units.Quantity
            The u-coordinates [astropy.units.meter]
        vcoords: astropy.untis.Quantity
            The v-coordinates [astropy.units.meter]
        """
        ucoords = self.get_data("oi_vis", "ucoord")*u.m
        vcoords = self.get_data("oi_vis", "vcoord")*u.m
        return ucoords.squeeze(), vcoords.squeeze()

    # TODO: Write test for this
    def get_uvcoords(self) -> Quantity:
        """Fetches the u, v coordinates from the (.fits)-files, merges them and gives the
        quantities the proper units

        Returns
        -------
        uvcoords: astropy.units.Quantity
            The (u, v)-coordinates [astropy.units.m]
        """
        return np.stack((self.get_split_uvcoords()), axis=1)

    # TODO: Write test for this
    def get_closures_phase_uvcoords_split(self) -> Quantity:
        """Fetches the (u1, v1), (u2, v2) coordinate of the closure phase triangles from
        the (.fits)-file, calculates the third (u3, v3) coordinate pair and then gives the
        quantities the proper units

        Returns
        -------
        u: Tuple[astropy.units.Quantity]
            The three u-coordinate pairs of the closure phase triangles
            [astropy.unit.meter]
        v: Tuple[astropy.units.Quantity]
            The three v-coordinate pairs of the closure phase triangles
            [astropy.unit.meter]
        """
        u1, v1 = self.get_data("oi_t3", "u1coord", "v1coord")
        u2, v2 = self.get_data("oi_t3", "u2coord", "v2coord")
        u3, v3 = -(u1+u2), -(v1+v2)
        return ([u1, u2, u3], [v1, v2, v3])*u.m

    def get_closures_phase_uvcoords(self) -> Quantity:
        ucoords, vcoords = self.get_closures_phase_uvcoords_split()
        uv_coords = [[], [], []]
        for i, ucoord in enumerate(ucoords):
            uv_coords[i] = np.stack((ucoord, vcoords[i]), axis=1)
        return u.Quantity(uv_coords)

    # TODO: Write test for this
    def get_baselines(self) -> Quantity:
        """Calculates the baselines from the uv coordinates

        Returns
        -------
        baselines: astropy.unit.Quantity
            The baselines [astropy.units.meter]
        """
        ucoords, vcoords = self.get_split_uvcoords()
        return np.sqrt(ucoords**2+vcoords**2)

    def get_closure_phases_baselines(self) -> Quantity:
        ucoords, vcoords = self.get_closures_phase_uvcoords_split()
        return np.sqrt(ucoords**2+vcoords**2)

    def get_visibilities(self) -> Quantity:
        """"Fetches the visibility data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities.

        Returns
        -------
        vis: astropy.units.Quantity
            The visibility of an observed object either [astropy.units.Jansky] or
            [astropy.units.dimensionless_unscaled]
        viserr: astropy.units.Quantity
            The error of the visibility of an observed object either
            [astropy.units.Jansky] or [astropy.units.dimensionless_unscaled]
        sta_indicies: astropy.units.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        vis, viserr = self.get_data("oi_vis", "visamp", "visamperr")

        if np.max(vis) > 1.:
            vis, viserr = map(lambda x: x*u.Jy, (vis, viserr))
        else:
            vis, viserr = map(lambda x: x*u.dimensionless_unscaled, (vis, viserr))
        return vis, viserr

    def get_visibilities_squared(self) -> Quantity:
        """Fetches the squared visibility data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities

        Returns
        ----------
        vis2: astropy.units.Quantity
            The squared visibility of an observed object
            [astropy.units.dimensionless_unscaled]
        vis2err: astropy.units.Quantity
            The error of the squared visibility of an observed object
            [astropy.units.dimensionless_unscaled]
        sta_indicies: astropy.units.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        vis2, vis2err = self.get_data("oi_vis2", "vis2data", "vis2err")
        return list(map(lambda x: x*u.dimensionless_unscaled, (vis2, vis2err)))

    def get_closure_phases(self) -> Quantity:
        """Fetches the closure phase data, its error and the sta_indicies from the
        (.fits)-file and gives the proper units to the quantities

        Returns
        ----------
        cphases: u.Quantity
            The closure phases of an observed object [astropy.units.degree]
        cphaseserr: u.Quantity
            The error of the closure phases of an observed object
            [astropy.units.degree]
        sta_indicies: u.Quantity
            The station indicies of the telescopes used
            [astropy.units.dimensionless_unscaled]
        """
        cphases, cphaseserr = self.get_data("oi_t3", "t3phi", "t3phierr")
        return list(map(lambda x: x*u.deg, (cphases, cphaseserr)))

    def get_flux(self) -> Quantity:
        """Fetches the (total) flux data, its error from the (.fits)-file and gives the
        proper units to the quantities

        Returns
        ----------
        flux: u.Quantity
            The (total) flux of an observed object [astropy.units.Jansky]
        fluxerr: u.Quantity
            The error of the (total) flux of an observed object [astropy.units.Jansky]
        """
        # TODO: Check how to handle if there is additional flux data -> Maybe only for one
        # dataset the flux
        if self.flux_file:
            return self._get_flux_file_data()
        else:
            return list(map(lambda x: x*u.Jy,
                            self.get_data("oi_flux", "fluxdata", "fluxerr")))

    def get_wavelength_solution(self) -> Quantity:
        """Fetches the wavelength solution from the (.fits)-file and gives the
        proper units to the quantities

        Returns
        ----------
        flux: astropy.units.Quantity
            The wavelength solution of the MATISSE instrument [astropy.units.micrometer]
        """
        return (self.get_data("oi_wavelength", "eff_wave")[0]*u.m).to(u.um)

    def get_visibilities4wavelength(self, wavelength_indices:\
                                    Union[List, np.ndarray]) -> Quantity:
        """Fetches the visdata(amp/phase)/correlated fluxes for one specific wavelength

        Parameters
        ----------
        wavelength_indices: List | numpy.ndarray
            The indicies of the wavelength solution

        Returns
        --------
        visamp4wavelength: astropy.units.Quantity
            The visamps for a specific wavelength either [astropy.units.Jansky] or
            [astropy.units.dimensionless_unscaled]
        visamperr4wavelength: astropy.units.Quantity
            The visamperrs for a specific wavelength either [astropy.units.Jansky] or
            [astropy.units.dimensionless_unscaled]
        """
        # FIXME: Is this ordering done correctly?? Check!
        visdata = self.get_visibilities()
        return self.get_data_for_wavelength(visdata, wavelength_indices)

    def get_visibilities_squared4wavelength(self, wavelength_indices:\
                                            Union[List, np.ndarray]) -> np.ndarray:
        """Fetches the vis2data for one specific wavelength

        Parameters
        ----------
        wavelength_indicies: List | numpy.ndarray
            The indicies of the wavelength solution

        Returns
        --------
        vis2data4wl: astropy.units.Quantity
            The vis2data for a specific wavelength [astropy.units.dimensionless_unscaled]
        vis2err4wl: astropy.units.Quantity
            The vis2err for a specific wavelength [astropy.units.dimensionless_unscaled]
        """
        vis2data = self.get_visibilities_squared()
        return self.get_data_for_wavelength(vis2data, wavelength_indices)

    def get_closure_phases4wavelength(self, wavelength_indices:\
                                      Union[List, np.ndarray]) -> Quantity:
        """Fetches the closure phases for one specific wavelength

        Parameters
        ----------
        wavelength_indicies: List
            The indicies of the wavelength solution

        Returns
        -------
        cphases4wl: astropy.units.Quantity
            The closure phase for a specific wavelength [astropy.units.degree]
        cphaseserr4wl: astropy.units.Quantity
            The closure phase error for a specific wavelength [astropy.units.degree]
        """
        cphasesdata = self.get_closure_phases()
        return self.get_data_for_wavelength(cphasesdata, wavelength_indices)

    def get_flux4wavelength(self, wavelength_indices: List) -> Quantity:
        """Fetches the flux for a specific wavelength

        Parameters
        ----------
        wavelength_indicies: List
            The indicies of the wavelength solution

        Returns
        -------
        wavelength_specific_flux: astropy.units.Quantity
            The flux for a specific wavelength [astropy.units.Jansky]
        wavelength_specific_fluxerr: astropy.units.Quantity
            The flux error for a specific wavelength [astropy.units.Jansky]
        """
        return self.get_data_for_wavelength(self.get_flux(), wavelength_indices)


if __name__ == "__main__":
    readout = ReadoutFits("../../../data/tests/test.fits")
    print(readout.get_closures_phase_uvcoords()[0])
    print(readout.get_closures_phase_uvcoords_split())

