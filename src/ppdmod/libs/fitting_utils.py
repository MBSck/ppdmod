from typing import Optional, List

import numpy as np
import astropy.units as u

from tqdm import tqdm
from astropy.units import Quantity

from .data_prep import DataHandler
from .combined_model import CombinedModel
from .fourier import FastFourierTransform


# def reformat_components_to_priors():
    # """Formats priors from the model components """
    # self._priors.clear()
    # self._labels.clear()
    # if self.model_components:
        # if self.geometric_priors is not None:
            # self._priors.extend([prior.value.tolist() for prior in self.geometric_priors])
            # self._labels.extend(["axis_ratio", "pa"])

        # if self.modulation_priors is not None:
            # self._priors.extend([prior.value.tolist() for prior in self.modulation_priors])
            # self._labels.extend(["mod_amp", "mod_angle"])

        # if self.disc_priors is not None:
            # if np.any(self.disc_priors.q.value):
                # self._priors.append(self.disc_priors.q)
                # self._labels.append("q")
            # if np.any(self.disc_priors.p.value):
                # self._priors.append(self.disc_priors.p)
                # self._labels.append("p")

        # # TODO: Add all possibilities here with the geometric params
        # for component in self.model_components:
            # if component.component == "ring":
                # # TODO: Implement geometric priors adding here
                # if self.geometric_priors is not None:
                    # ...
                # if component.priors.inner_radius.value.any():
                    # self._priors.append(component.priors.inner_radius.value.tolist())
                    # self._labels.append(f"{component.name}:ring:inner_radius")
                # if component.priors.outer_radius.value.any():
                    # self._priors.append(component.priors.outer_radius.value.tolist())
                    # self._labels.append(f"{component.name}:ring:outer_radius")

        # if self.lnf_priors is not None:
            # lnf_priors = self.lnf_priors
        # else:
            # lnf_priors = [0., 0.]

        # self._priors.append(lnf_priors)
        # self._labels.append("lnf")

    # else:
        # raise ValueError("No model components have been added yet!")

# def reformat_theta_to_components(self, theta: List[float]):
    # """Sets the model component list anew from the input theta"""
    # # TODO: Figure out way to do this during the modelling
    # # TODO: Make this into independent function
    # model_components = []
    # theta_dict = dict(zip(self.labels, theta))
    # for component in self.model_components:
        # if component.component == "delta":
            # model_components.append(component)
            # break

    # self.model_components.clear()
    # if "axis_ratio" in theta_dict:
        # self.geometric_params = [theta_dict["axis_ratio"], theta_dict["pa"]]
    # if "mod_angle" in theta_dict:
        # self.modulation_params = [theta_dict["mod_angle"], theta_dict["mod_amp"]]

    # # TODO: This might need a rework for more complex models
    # if ("q" and "p") in theta_dict:
        # self.disc_params = [theta_dict["q"], theta_dict["p"]]
    # else:
        # if "q" in theta_dict:
            # self.disc_params = [theta_dict["q"], 0.]
        # else:
            # self.disc_params = [0., theta_dict["p"]]
    # if "lnf" in theta_dict:
        # self.lnf = theta_dict["lnf"]

    # component_params_dict = {}
    # for key, value in theta_dict.items():
        # if ":" not in key:
            # continue
        # name, component_name, param_name = key.split(":")
        # if not name in component_params_dict:
            # component_params_dict[name] = {}
        # if not "params" in component_params_dict[name]:
            # component_params_dict[name]["params"] = {}
        # component_params_dict[name]["component"] = component_name
        # if component_name == "ring":
            # # TODO: Find a way to get around this parameter checking
            # if self.geometric_params is not None:
                # component_params_dict[name]["params"]["axis_ratio"] =\
                    # theta_dict["axis_ratio"]
                # component_params_dict[name]["params"]["pa"] = theta_dict["pa"]
            # if self.modulation_params is not None:
                # mod_params = [theta_dict["mod_amp"], theta_dict["mod_angle"]]
                # component_params_dict[name]["mod_params"] = mod_params
        # component_params_dict[name]["params"][param_name] = value

    # for name, values in component_params_dict.items():
        # mod_params = None
        # if values["component"] == "ring":
            # params = [value for value in values["params"].values()]
            # if "outer_radius" not in values["params"]:
                # params.append(0.)
            # if "mod_params" in values:
                # mod_params = [value for value in values["mod_params"]]
            # component = make_ring_component(name, params=params,
                                            # mod_params=mod_params)
        # model_components.append(component)
    # self.model_components = model_components.copy()


def loop_model(model: CombinedModel, data: DataHandler,
               wavelength: Quantity, rfourier: Optional[bool] = False):
    """"""
    image = model.eval_flux(wavelength)
    total_flux = model.eval_total_flux(wavelength).value
    total_flux_arr = [total_flux]
    total_flux_arr = np.array([total_flux for _ in range(data.corr_fluxes.shape[1] // 6)])
    fourier = FastFourierTransform(image, wavelength,
                                        data.pixel_size, data.zero_padding_order)
    corr_flux_arr, cphases_arr = fourier.get_uv2fft2(data.uv_coords, data.uv_coords_cphase)
    if rfourier:
        return total_flux_arr, corr_flux_arr, cphases_arr, fourier
    else:
        return total_flux_arr, corr_flux_arr, cphases_arr


# TODO: Write tests for this function
# TODO: Check if works as thought
def calculate_model(theta: np.ndarray, data: DataHandler,
                    rfourier: Optional[bool] = False, debug: Optional[bool] = False):
    """"""
    data.reformat_theta_to_components(theta)
    model = CombinedModel(data.fixed_params, data.disc_params,
                          data.wavelengths, data.geometric_params,
                          data.modulation_params)
    model.tau = data.tau_initial
    for component in data.model_components:
        model.add_component(component)

    total_flux_mod_chromatic, corr_flux_mod_chromatic, cphases_mod_chromatic_data =\
        [], [], []
    if debug:
        for wavelength in tqdm(data.wavelengths, "Calculating polychromatic model..."):
            model_data = loop_model(model, data, wavelength, rfourier)
            total_flux_mod_chromatic.append(model_data[0])
            corr_flux_mod_chromatic.append(model_data[1])
            cphases_mod_chromatic_data.append(model_data[2])
    else:
        for wavelength in data.wavelengths:
            model_data = loop_model(model, data, wavelength, rfourier)
            total_flux_mod_chromatic.append(model_data[0])
            corr_flux_mod_chromatic.append(model_data[1])
            cphases_mod_chromatic_data.append(model_data[2])

    if rfourier:
        return total_flux_mod_chromatic*u.Jy, corr_flux_mod_chromatic*u.Jy,\
            cphases_mod_chromatic_data*u.deg, model_data[-1]
    return total_flux_mod_chromatic*u.Jy, corr_flux_mod_chromatic*u.Jy,\
        cphases_mod_chromatic_data*u.deg


def lnlike(theta: np.ndarray, data: DataHandler) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.


    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    data: DataHandler

    Returns
    -------
    float
        The goodness of the fitted model (will be minimised)
    """
    lnf = theta[-1]
    total_flux_mod, corr_flux_mod, cphases_mod = calculate_model(theta[:-1], data)

    if data.fit_total_flux:
        total_flux_chi_sq = chi_sq(data.total_fluxes,
                                   data.total_fluxes_error, total_flux_mod, lnf)
    else:
        total_flux_chi_sq = 0

    corr_flux_chi_sq = chi_sq(data.corr_fluxes, data.corr_fluxes_error, corr_flux_mod, lnf)
    if data.fit_cphases:
        cphases_chi_sq = chi_sq(data.cphases, data.cphases_error, cphases_mod, lnf)
    else:
        cphases_chi_sq = 0
    return np.array(total_flux_chi_sq+corr_flux_chi_sq+cphases_chi_sq)


def lnprior(theta: np.ndarray, priors: List[List[float]]) -> float:
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    priors: List[List[float]]
        A list containing all the priors' bounds

    Returns
    -------
    float
        Return-code 0.0 for within bounds and -np.inf for out of bound priors
    """
    for index, (upper_bound, lower_bound) in enumerate(priors):
        if not lower_bound < theta[index] < upper_bound:
            return -np.inf
    return 0.


def lnprob(theta: np.ndarray, data: DataHandler) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model

    Returns
    -------
    float
        The minimisation value or -np.inf if it fails
    """
    return lnlike(theta, data) if np.isfinite(lnprior(theta, data.priors)) else -np.inf


def chi_sq(real_data: Quantity, data_error: Quantity,
           data_model: Quantity, lnf: float) -> float:
    """The chi square minimisation

    Parameters
    ----------
    real_data: astropy.units.Quantity
    data_error: astropy.units.Quantity
    data_model: astropy.units.Quantity
    lnf: float, optional

    Returns
    -------
    float
    """
    inv_sigma_squared = 1./np.sum(data_error.value**2 +
                                  data_model.value**2*np.exp(2*lnf))
    return -0.5*np.sum((real_data.value-data_model.value)**2*inv_sigma_squared
                       - np.log(inv_sigma_squared))


if __name__ == "__main__":
    ...
