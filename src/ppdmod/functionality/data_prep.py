import numpy as np
import astropy.units as u

from pathlib import Path
from astropy.units import Quantity
from typing import List, Optional, Callable

from .readout import ReadoutFits
from .utils import IterNamespace, make_delta_component, make_ring_component

# TODO: For this class implement different methods of polychromatic fitting. At a later
# time
# TODO: This will for the start just fit either the L- or N-band and not both at the
# same time, fix this later
# TODO: Check if this works for the uv-coords as well, if not make another merge_func
class DataHandler:
    """This class handles all the data that is used for the fitting process, the observed
    data as well as the data created by the modelling process"""
    def __init__(self, fits_files: List[Path],
                 wavelengths: List[float],
                 wavelength_window_sizes: Optional[List[float]] = [0.2],
                 flux_file: Optional[Path] = None) -> None:
        """Initialises the class"""
        self.fits_files, self.flux_file = fits_files, flux_file
        self._wavelengths = wavelengths
        self._wavelength_window_sizes = wavelength_window_sizes
        self.readout_files = [ReadoutFits(fits_file, self.flux_file)\
                              for fits_file in self.fits_files]
        self.getter_function_dictionary = {"vis": "get_visibilities4wavelength",
                                           "vis2": "get_visibilities_squared4wavelength",
                                           "cphases": "get_closure_phases4wavelength",
                                           "flux": "get_flux4wavelength"}
        # NOTE: This only works if the wl_solution stays the same for all files
        self.wl_ind = self.readout_files[0].\
            get_wavelength_indices(self._wavelengths, self._wavelength_window_sizes)

        self.model_components = []
        self._geometric_priors, self._modulation_priors = None, None
        self._geometric_params, self._modulation_params = None, None
        self._priors, self._labels = [], []

    def __repr__(self):
        """The DataHandler class' representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(i) for i in self.readout_files])}\n and polychromatic"\
            f"data of {self.wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    def __str__(self):
        """The DataHandler class' string representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(i) for i in self.readout_files])}\n and polychromatic"\
            f"data of {self.wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    @property
    def wavelengths(self):
        if not isinstance(self._wavelengths, u.Quantity):
            return self._wavelengths*u.um
        elif self._wavelengths.unit != u.um:
            raise ValueError
        else:
            return self._wavelength_window_sizes

    @property
    def wavelength_window_sizes(self):
        if not isinstance(self._wavelength_window_sizes, u.Quantity):
            return self._wavelength_window_sizes*u.um
        elif self._wavelength_window_sizes.unit != u.um:
            raise ValueError
        else:
            return self._wavelength_window_sizes

    @property
    def geometric_priors(self):
        """Gets the geometric priors"""
        return self._geometric_priors

    @geometric_priors.setter
    def geometric_priors(self, value):
        units = [u.dimensionless_unscaled, u.deg]
        labels = ["axis_ratio", "pa"]
        self._geometric_priors = _make_priors(value, units, labels)

    @property
    def geometric_params(self):
        return self._geometric_params

    @geometric_params.setter
    def geometric_params(self, value: List[float]):
        units = [u.dimensionless_unscaled, u.deg]
        labels = ["axis_ratio", "pa"]
        self._geometric_params = _make_params(value, units, labels)

    @property
    def modulation_priors(self):
        return self._modulation_priors

    @modulation_priors.setter
    def modulation_priors(self, value):
        units = [u.deg, u.dimensionless_unscaled]
        labels = ["mod_angle", "mod_amp"]
        self.modulation_priors = _make_priors(value, units, labels)

    @property
    def modulation_params(self):
        return self._modulation_params

    @modulation_params.setter
    def modulation_params(self, value):
        units = [u.deg, u.dimensionless_unscaled]
        labels = ["mod_angle", "mod_amp"]
        self._modulation_params = _make_params(value, units, labels)

    # TODO: Implement this for more components at a later time
    # TODO: Implement this for different orientations and modulations
    def _reformat_components_to_priors(self):
        """Formats priors from the model components """
        self._priors, self._labels = [], []
        if self.model_components:
            if self.geometric_priors is not None:
                self._priors.extend(self.geometric_priors.value)
                self._labels.extend(["axis_ratio", "pa"])

            if self.modulation_priors is not None:
                self._priors.extend(self.modulation_priors.value)
                self._labels.extend(["mod_amp", "mod_angle"])

            if self.disc_params is not None:
                self._priors.extend(self.disc_params.value)
                self._labels.extend(["q", "p"])

            # TODO: Add all possibilities here with the geometric params
            for component in self.model_components:
                if component.component == "ring":
                    if self.geometric_priors is not None:
                        if not component.priors.inner_radius.all():
                            self._priors.append(component.priors.inner_radius.value)
                            self._labels.append(f"{component.name}:ring:inner_radius")
                        if not component.priors.outer_radius.all():
                            self._priors.append(component.priors.outer_radius.value)
                            self._labels.append(f"{component.name}:ring:outer_radius")
        else:
            raise ValueError("No model components have been added yet!")

    def _reformat_theta_to_components(self, theta: List[float]):
        theta_dict = dict(zip(self._labels, theta))
        model_components = []
        if self.geometric_priors is not None:
            self.geometric_params = [theta_dict["axis_ratio"], theta_dict["pa"]]
        if self.modulation_priors is not None:
            self.modulation_params = [theta_dict["mod_angle"], theta_dict["mod_amp"]]
        for component in self.model_components:
            if component.component == "delta":
                model_components.append(make_delta_component("star"))

        # FIXME: Think of a way to implement this
        components_dic = {}
        for key, value in theta_dict.items():
            name, component_name, param_name = key.split(":")
            components_dic[f"{name}:{component_name}"] = {param_name: value}

        raise ValueError("No model components have been added yet!")

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

    def add_model_component(self, model_component: IterNamespace):
        self.model_components.append(model_component)


if __name__ == "__main__":
    fits_files = ["../../../assets/data/Test_model.fits"]*2
    wavelengths = [8.5, 10.0]
    data = DataHandler(fits_files, wavelengths)
    print(data._merge_data("vis"))

