from astropy.time.core import enum
import numpy as np
import astropy.units as u
import warnings

from pathlib import Path
from astropy.units import Quantity
from typing import List, Optional, Callable

from .readout import ReadoutFits
from .utils import IterNamespace, _make_params, make_delta_component,\
    make_ring_component, make_fixed_params, _make_priors, make_inital_guess_from_priors

# TODO: For this class implement different methods of polychromatic fitting. At a later
# time
# TODO: This will for the start just fit either the L- or N-band and not both at the
# same time, fix this later
# TODO: Check if this works for the uv-coords as well, if not make another merge_func
class DataHandler:
    """This class handles all the data that is used for the fitting process, the observed
    data as well as the data created by the modelling process"""
    def __init__(self, fits_files: List[Path],
                 wavelengths: List[Quantity],
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
                                           "flux": "get_flux4wavelength",
                                           "uvcoords": "get_uvcoords",
                                           "uvcoords_cphase": "get_closures_phase_uvcoords",
                                           "telescope": "get_telescope_information",
                                           "baselines": "get_baselines",
                                           "baselines_cphase": "get_closure_phases_baselines"}
        # NOTE: This only works if the wl_solution stays the same for all files
        self.wl_ind = self.readout_files[0].\
            get_wavelength_indices(self.wavelengths, self.wavelength_window_sizes)

        self.model_components = []
        self.fixed_params = None
        self.zero_padding_order = 1
        self._disc_priors, self._geometric_priors, self._modulation_priors =\
            None, None, None
        self._disc_params, self._geometric_params, self._modulation_params =\
            None, None, None
        self._tau_initial = None
        self._priors, self._labels = [], []
        self._mcmc = None

        self.total_fluxes, self.total_fluxes_error = self._merge_data("flux")
        self.total_fluxes_sigma_squared = self._get_sigma_square("flux")
        self.corr_fluxes, self.corr_fluxes_error = self._merge_data("vis")
        self.corr_fluxes_sigma_squared = self._get_sigma_square("vis")
        self.cphases, self.cphases_error = self._merge_data("cphases")
        self.cphases_sigma_squared = self._get_sigma_square("cphases")

        self.uv_coords = self._merge_simple_data("uvcoords")
        self.uv_coords_cphase = self._merge_simple_data("uvcoords_cphase")
        self.telescope_info = self._merge_simple_data("telescope")
        self.baselines = self._merge_simple_data("baselines")
        self.baselines_cphase = self._merge_simple_data("baselines_cphase")

    def __repr__(self):
        """The DataHandler class' representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(file) for file in self.readout_files])}\n and polychromatic"\
            f"data of {self.wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    def __str__(self):
        """The DataHandler class' string representation"""
        return f"DataHandler contains the information of the ReadoutFits:"\
            f"\n{', '.join([str(file) for file in self.readout_files])}\n and polychromatic"\
            f"data of {self.wavelengths} with the windows "\
            f"{self.wavelength_window_sizes}"

    @property
    def tau_initial(self):
        if self._tau_initial is None:
            raise ValueError("The value for tau has not been set yet!")
        return self._tau_initial

    @tau_initial.setter
    def tau_initial(self, value):
        if not isinstance(value, u.Quantity):
            self._tau_initial = value*u.dimensionless_unscaled
        elif value.unit != u.dimensionless_unscaled:
            raise IOError(f"Wrong unit has been input for tau initial. Needs to"\
                          f" be in [astropy.units.dimensionless_unscaled] or unitless!")

    # TODO: Finish this function and get the poper baselines
    @property
    def longest_baselines(self):
        # longest_baselines = self.baselines_cphase.flatten().copy()
        # longest_baselines.sort()
        return self.baselines_cphase[0]

    @property
    def pixel_scaling(self):
        if self.fixed_params is not None:
            return self.fixed_params.fov/self.fixed_params.image_size
        else:
            raise ValueError("Fixed params have to be specified to get pixel_scaling!")

    @property
    def initial(self):
        if self.priors is not None:
            return make_inital_guess_from_priors(self.priors)
        else:
            raise ValueError("Priors have to be set before making initial guess!")

    @property
    def wavelengths(self):
        if not isinstance(self._wavelengths, u.Quantity):
            return self._wavelengths*u.um
        elif self._wavelengths.unit != u.um:
            raise ValueError("Wrong input unit for the wavelengths!"\
                             " Needs to be in [astropy.units.um]")
        else:
            return self._wavelengths

    @property
    def wavelength_window_sizes(self):
        if not isinstance(self._wavelength_window_sizes, u.Quantity):
            self._wavelength_window_sizes *= u.um
        elif self._wavelength_window_sizes.unit != u.um:
            raise ValueError("Wrong input unit for the wavelength windows!"\
                             " Needs to be in [astropy.units.um]")

        if len(self._wavelength_window_sizes) == 1\
                and self.wavelengths.shape[0] != len(self._wavelength_window_sizes):
            return np.repeat(self._wavelength_window_sizes, self.wavelengths.shape[0])
        elif len(self._wavelength_window_sizes) != self.wavelengths.shape[0]:
            raise IOError("The specified wavelength windows have be the same length"\
                          " as the selected wavelength list!")
        else:
            return self._wavelength_window_sizes

    @property
    def mcmc(self):
        if self._mcmc is None:
            warnings.warn("The hyperparams for the mcmc-fitting have not been set."\
                          " Defaults to None!")
        return self._mcmc

    @mcmc.setter
    def mcmc(self, value: List[float]):
        key = ["initial", "ndim", "nwalkers", "nburn", "niter", "frac"]
        initial = self.initial
        value.insert(0, initial)
        value.insert(1, initial.shape[0])
        self._mcmc = IterNamespace(**dict(zip(key, value)))

    @property
    def priors(self):
        if not self._priors:
            self.reformat_components_to_priors()
        return self._priors

    @property
    def labels(self):
        if not self._labels:
            self.reformat_components_to_priors()
        return self._labels

    @property
    def disc_priors(self):
        """Gets the geometric priors"""
        return self._disc_priors

    @disc_priors.setter
    def disc_priors(self, value):
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled]
        labels = ["q", "p"]
        self._disc_priors = _make_priors(value, units, labels)

    @property
    def disc_params(self):
        return self._disc_params

    @disc_params.setter
    def disc_params(self, value: List[float]):
        units = [u.dimensionless_unscaled, u.dimensionless_unscaled]
        labels = ["q", "p"]
        self._disc_params = _make_params(value, units, labels)

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
        self._modulation_priors = _make_priors(value, units, labels)

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
    def reformat_components_to_priors(self):
        """Formats priors from the model components """
        self._priors, self._labels = [], []
        if self.model_components:
            if self.geometric_priors is not None:
                self._priors.extend([prior.value.tolist() for prior in self.geometric_priors])
                self._labels.extend(["axis_ratio", "pa"])

            if self.modulation_priors is not None:
                self._priors.extend([prior.value.tolist() for prior in self.modulation_priors])
                self._labels.extend(["mod_amp", "mod_angle"])

            if self.disc_priors is not None:
                self._priors.extend([prior.value.tolist() for prior in self.disc_priors])
                self._labels.extend(["q", "p"])

            # TODO: Add all possibilities here with the geometric params
            for component in self.model_components:
                if component.component == "ring":
                    # TODO: Implement geometric priors adding here
                    if self.geometric_priors is not None:
                        ...
                    if component.priors.inner_radius.value.any():
                        self._priors.append(component.priors.inner_radius.value.tolist())
                        self._labels.append(f"{component.name}:ring:inner_radius")
                    if component.priors.outer_radius.value.any():
                        self._priors.append(component.priors.outer_radius.value.tolist())
                        self._labels.append(f"{component.name}:ring:outer_radius")
        else:
            raise ValueError("No model components have been added yet!")

    def reformat_theta_to_components(self, theta: List[float]):
        """Sets the model component list anew from the input theta"""
        model_components = []
        theta_dict = dict(zip(self.labels, theta))
        for component in self.model_components:
            if component.component == "delta":
                model_components.append(component)
                break

        self.model_components = []
        if "axis_ratio" in theta_dict:
            self.geometric_params = [theta_dict["axis_ratio"], theta_dict["pa"]]
        if "mod_angle" in theta_dict:
            self.modulation_params = [theta_dict["mod_angle"], theta_dict["mod_amp"]]
        if "q" in theta_dict:
            self.disc_params = [theta_dict["q"], theta_dict["p"]]

        component_params_dict = {}
        for key, value in theta_dict.items():
            if ":" not in key:
                continue
            name, component_name, param_name = key.split(":")
            if not name in component_params_dict:
                component_params_dict[name] = {}
            if not "params" in component_params_dict[name]:
                component_params_dict[name]["params"] = {}
            component_params_dict[name]["component"] = component_name
            if component_name == "ring":
                if self.geometric_params is not None:
                    component_params_dict[name]["params"]["axis_ratio"] =\
                        theta_dict["axis_ratio"]
                    component_params_dict[name]["params"]["pa"] = theta_dict["pa"]
                if self.modulation_params is not None:
                    mod_params = [theta_dict["mod_amp"], theta_dict["mod_angle"]]
                    component_params_dict[name]["mod_params"] = mod_params
            component_params_dict[name]["params"][param_name] = value

        for name, values in component_params_dict.items():
            if values["component"] == "ring":
                params = [value for value in values["params"].values()]
                if "outer_radius" not in values["params"]:
                    params.append(0.)
                mod_params = [value for value in values["mod_params"]]
                component = make_ring_component(name, params=params,
                                                mod_params=mod_params)
            model_components.append(component)

        self.model_components = model_components.copy()

    def _get_data_type_function(self, readout_file: Callable,
                                data_keyword: str) -> Callable:
        """This gets a method, to get a certain datatype, to be called from the
        ReadoutFits class via a keyword provided

        Parameters
        ----------
        readout_file: Callable
            The class that is to be checked for the method
        data_type_keyword: str
            A keyword from "vis", "vis2", "cphases" or "flux", "uvcoords",
            "uvcoords_cphase", "telescope", "baselines" or "baselines_cphase"
            that is used to get the specified function

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
        if data_type_keyword in ["uvcoords", "uvcoords_cphase",
                                 "telescope", "baselines", "baselines_cphase"]:
            raise IOError("Use the '_merge_simple_data' function for these datatypes")
        for i, readout_file in enumerate(self.readout_files):
            getter_func = self._get_data_type_function(readout_file, data_type_keyword)
            data = getter_func(self.wl_ind)
            if i == 0 and len(self.readout_files) > 1:
                getter_func_next = self._get_data_type_function(self.readout_files[i+1],
                                                                data_type_keyword)
                data_next = getter_func_next(self.wl_ind)
                merged_data = self._iterate_over_data_arrays(data, data_next)
            elif len(self.readout_files) == 1:
                merged_data = data
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

    def _merge_simple_data(self, data_type_keyword):
        """Merges simple data, like the 'uvcoords', 'uvcoords_cphase', 'telescope',
        'baselines' or 'baselines_cphase' data"""
        merged_data = np.array([])
        if data_type_keyword in ["vis", "vis2", "cphases", "flux"]:
            raise IOError("Use the '_merge_data' function for these datatypes")
        for readout_file in self.readout_files:
            getter_func = self._get_data_type_function(readout_file, data_type_keyword)
            data = getter_func()
            if merged_data.size == 0:
                merged_data = data
            else:
                if data_type_keyword == "uvcoords_cphase":
                    temp_data_lst = [[], [], []]
                    for i, uv_coords in enumerate(data):
                            temp_data_lst[i] = np.concatenate((merged_data[i], uv_coords))
                    merged_data = temp_data_lst.copy()
                elif data_type_keyword == "baselines_cphase":
                    temp_data_lst = [[], [], []]
                    for i, baseline in enumerate(data):
                        temp_data_lst[i] = np.concatenate((merged_data[i], baseline))
                    merged_data = temp_data_lst.copy()
                elif data_type_keyword == "telescope":
                    temp_data_lst = [[], [], [], []]
                    for i, dataset in enumerate(data):
                        temp_data_lst[i] = np.concatenate((merged_data[i], dataset))
                    merged_data = temp_data_lst.copy()
                else:
                    merged_data = np.concatenate((merged_data, data))

        if not data_type_keyword == "telescope":
            return u.Quantity(merged_data)
        else:
            return merged_data

    def add_model_component(self, model_component: IterNamespace):
        self.model_components.append(model_component)


if __name__ == "__main__":
    fits_files = ["../../../data/tests/test.fits"]*2
    flux_file = "../../../data/tests/HD_142666_timmi2.txt"
    wavelengths = [8.5, 10.0]
    theta = [0.5, 145, 1., 35, 0.5, 0.05, 3., 5., 7.]
    data = DataHandler(fits_files, wavelengths, flux_file=flux_file)
    print(data.total_fluxes)
    complete_ring = make_ring_component("inner_ring",
                                        [[0., 0.], [0., 0.], [1., 6.], [0., 0.]])
    delta_component = make_delta_component("star")
    data.add_model_component(delta_component)
    data.add_model_component(complete_ring)
    data.mcmc = [100, 5, 5, 1e-4]
    print(*data.mcmc)
    # data._labels = ["axis_ratio", "pa", "mod_amp", "mod_angle", "q", "p",
                   # "inner:ring:inner_radius", "inner:ring:outer_radius",
                   # "outer:ring:inner_radius"]
    # # print(data.model_components)
    # data.reformat_theta_to_components(theta)
    # # print(data.disc_params.q)
    # # print(data.model_components)
    # print(data.uv_coords)

    # data2 = DataHandler(fits_files, wavelengths)
    # data2.geometric_priors = [[0., 1.], [0, 180]]
    # data2.modulation_priors = [[0., 1.], [0, 360]]
    # data2.disc_priors = [[0., 1.], [0., 1.]]
    # data2.zero_padding_order = 2
    # data2.fixed_params = make_fixed_params(30, 128, 1500, 7900, 140, 19, 1)
    # complete_ring = make_ring_component("inner_ring",
                                        # [[0., 0.], [0., 0.], [3., 5.], [0., 0.]])
    # delta_component = make_delta_component("star")
    # data2.add_model_component(delta_component)
    # data2.add_model_component(complete_ring)
    # data2.reformat_components_to_priors()
    # # print(data2.total_fluxes.value)
    # # print(data2.corr_fluxes.value)
    # # print(data2.cphases_sigma_squared.value)

    # values = [32, 1000, 5000, 1e-4]
    # data2.mcmc = values
    # # print(data2.pixel_scaling)
