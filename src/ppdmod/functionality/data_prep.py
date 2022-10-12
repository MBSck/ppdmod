import os
import numpy as np
import astropy.units as u

from pathlib import Path
from astropy.units import Quantity
from typing import Tuple, Dict, List, Optional, Union, Callable

from .readout import ReadoutFits

# TODO: For this class implement different methods of polychromatic fitting. At a later
# time
# TODO: This will for the start just fit either the L- or N-band and not both at the
# same time, fix this later
# TODO: Check if this works for the uv-coords as well, if not make another merge_func
class DataHandler:
    """This class handles all the data that is used for the fitting process, the observed
    data as well as the data created by the modelling process"""
    def __init__(self, fits_files: List[Path],
                 selected_wavelengths: List[float],
                 wavelength_window_sizes: Optional[List[float]] = [0.2],
                 flux_file: Optional[Path] = None) -> None:
        """Initialises the class"""
        self.fits_files, self.flux_file = fits_files, flux_file
        self.selected_wavelengths = selected_wavelengths
        self.wavelength_window_sizes = wavelength_window_sizes
        self.readout_files = [ReadoutFits(fits_file, self.flux_file)\
                              for fits_file in self.fits_files]
        self.getter_function_dictionary = {"vis": "get_visibilities4wavelength",
                                           "vis2": "get_visibilities_squared4wavelength",
                                           "cphases": "get_closure_phases4wavelength",
                                           "flux": "get_flux4wavelength"}
        # NOTE: This only works if the wl_solution stays the same for all files
        self.wl_ind = self.readout_files[0].\
            get_wavelength_indices(self.selected_wavelengths, self.wavelength_window_sizes)

    def __repr__(self):
        """The DataHandler class' representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(i) for i in self.readout_files])}\n and polychromatic"\
            f"data of {self.selected_wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    def __str__(self):
        """The DataHandler class' string representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(i) for i in self.readout_files])}\n and polychromatic"\
            f"data of {self.selected_wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    def _get_data_type_function(self, readout_file: Callable,
                               data_keyword: str) -> Callable:
        """This gets a method, to get a certain datatype, to be called from the
        ReadoutFits class via a keyword provided

        Parameters
        ----------
        readout_file: Callable
            The class that is to be checked for the method
        data_type_keyword: str
            A keyword from "vis", "vis2", "cphases" or "flux" that is used to get the
            correct function

        Returns
        -------
        data_getter_function: Callable
        """
        return getattr(readout_file, self.getter_function_dictionary[data_keyword])

    def _iterate_over_data_arrays(self, data: List, data_other: List) -> List:
        """Iterates two arrays and merges them

        Parameters
        ----------
        data: List
            The first data to be merged
        data_other: List
            The other data that is to be merged

        Returns
        -------
        List
            The merged data
        """
        merged_data = [[], []]
        for i, dataset in enumerate(data.copy()):
            for j, polychromatic_data in enumerate(dataset):
                unit = polychromatic_data.unit
                if not np.any(merged_data[i]):
                    merged_data[i] = np.zeros((dataset.shape[0],
                                              polychromatic_data.shape[0]+\
                                              data_other[i][j].shape[0]))
                merged_data[i][j] = np.append(polychromatic_data, data_other[i][j])

            merged_data[i] *= unit

        return merged_data

    # TODO: Check if this works for the uv-coords as well, if not make another merge_func
    def _merge_data(self, data_type_keyword: str) -> Quantity:
        """Fetches the data from the individual ReadoutFits classes for the selected
        wavelengths and then merges them into new longer arrays for the determined
        keyword.
        The new arrays are of the same shape as before, but extended by their length times
        the number of additional (.fits)-files that have been read

        Parameters
        ----------
        data_type_keyword: str
            A keyword from "vis", "vis2", "cphases" or "flux" that is used to get the
            correct function

        Returns
        -------
        merged_data: astropy.units.Quantity
            The merged datasets, extended by their own length times the amount of
            (.fits)-files passed to the DataHandler class
        """
        for i, readout_file in enumerate(self.readout_files):
            getter_func = self._get_data_type_function(readout_file, data_type_keyword)
            data = getter_func(self.wl_ind)
            if i == 0:
                getter_func_next = self._get_data_type_function(self.readout_files[i+1],
                                                                data_type_keyword)
                data_next = getter_func_next(self.wl_ind)
                merged_data = self._iterate_over_data_arrays(data, data_next)
            elif i == 1:
                continue
            else:
                merged_data = self._iterate_over_data_arrays(merged_data, data)

        return merged_data

    # TODO: Write function that combines flux and vis, but leave it alone for now
    def _get_sigma_square(self, data_type_keyword: str) -> Quantity:
        """Fetches the errors from the datatype and then squares them

        Parameters
        ----------
        data_type_keyword: str
            A keyword from "vis", "vis2", "cphases" or "flux" that is used to get the
            correct function

        Returns
        -------
        sigma_square: astropy.units.Quantity
            The squared error of the corresponding datatype
        """
        return self._merge_data(data_type_keyword)[1]**2

    # def do_model(parameters: np.ndarray, model_param_lst: List,
                            # uv_info_lst: List, vis_lst: List) -> np.ndarray:
        # """The model image, that is Fourier transformed for the fitting process

        # Parameters
        # ----------
        # theta: np.ndarray
        # model_param_lst: List
        # uv_info_lst: List
        # vis_lst: List

        # Returns
        # -------
        # amp: np.ndarray
            # Amplitudes of the model interpolated for the (u, v)-coordinates
        # cphases: np.ndarray
            # Closure phases of the model interpolated for the (u, v)-coordinates
        # """
        # # TODO: Work this into the class
        # model, pixel_size, sampling, wavelength,\
                # zero_padding_order, bb_params, _ = model_param_lst
        # uvcoords, u, v, t3phi_uvcoords = uv_info_lst
        # vis, vis2, intp = vis_lst

        # amp_lst, cphase_lst = [], []

        # for i in wavelength:
            # model_base = model(*bb_params, i)
            # model_flux = model_base.eval_model(theta, pixel_size, sampling)
            # fft = FFT(model_flux, i, model_base.pixel_scale,
                     # zero_padding_order)
            # amp, cphase, xycoords = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                                   # corr_flux=vis, vis2=vis2,
                                                   # intp=intp)
            # if len(amp) > 6:
                # flux_ind = np.where([i % 6 == 0 for i, o in
                                     # enumerate(amp)])[0].tolist()
                # amp = np.insert(amp, flux_ind, np.sum(model_flux))
            # else:
                # amp = np.insert(amp, 0, np.sum(model_flux))

            # amp_lst.append(amp)
            # cphase_lst.append(cphase)

        # return np.array(amp_lst), np.array(cphase_lst)


if __name__ == "__main__":
    ...

