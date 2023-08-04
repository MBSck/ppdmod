from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict

import astropy.units as u
import emcee
import numpy as np

from .custom_components import Star, AsymmetricSDGreyBodyContinuum
from .fft import interpolate_coordinates
from .model import Model
from .parameter import Parameter
from .options import OPTIONS
from .utils import execution_time


def chi_sq(data: u.quantity, error: u.quantity,
           model_data: u.quantity, lnf: float = 0) -> float:
    """the chi square minimisation.

    Parameters
    ----------
    data : numpy.ndarray
    error : numpy.ndarray
    model_data : numpy.ndarray
    lnf : float, optional

    Returns
    -------
    chi_sq : float
    """
    inv_sigma_squared = 1./np.sum(
        error**2 + model_data**2*np.exp(2*lnf))
    return -0.5*np.sum(
        (data-model_data)**2*inv_sigma_squared-np.log(inv_sigma_squared))


@execution_time
def lnprob(theta: np.ndarray) -> float:
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

    Notes
    -----
    If multiple models with the same params are required put them once into theta
    and then assign them with a dict, to then get them from the dict for the second
    component as well.
    """
    params = dict(zip(OPTIONS["model.params"].keys(), theta))

    # TODO: If more than one model component make all params variable.
    # Do this via an option that takes both the models and the params.
    for index, parameter in enumerate(OPTIONS["model.params"].values()):
        if parameter.free:
            if not parameter.min < theta[index] < parameter.max:
                return -np.inf

    star = Star(**OPTIONS["model.constant_params"])
    temp_grad = AsymmetricSDGreyBodyContinuum(
        **OPTIONS["model.constant_params"], **params)
    model = Model([star, temp_grad])

    fourier_transforms = {}
    for wavelength in OPTIONS["fit.wavelengths"]:
        fourier_transform = model.calculate_complex_visibility(wavelength)
        # total_flux = fourier_transform
        fourier_transforms[str(wavelength)] = {"ft": fourier_transform}
        #                                      "total_flux": total_flux}

    # TODO: Calculate the different observables here
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]
    total_chi_sq = 0
    for index, (corr_flux, corr_flux_err, cphase, cphase_err)\
            in enumerate(
                zip(corr_fluxes, corr_fluxes_err, cphases, cphases_err)):
        for wavelength in OPTIONS["fit.wavelengths"]:
            if str(wavelength) in fourier_transforms\
                    and str(wavelength) in corr_flux:
                fourier_transform = fourier_transforms[str(wavelength)]
                interpolated_fft = interpolate_coordinates(
                    fourier_transform["ft"],
                    fourier_transform["ft"].shape[0],
                    temp_grad.params["pixel_size"](),
                    OPTIONS["data.readouts"][index].ucoord,
                    OPTIONS["data.readouts"][index].vcoord,
                    wavelength)

                # total_flux_model = fourier_transform["total_flux"]
                corr_flux_model = np.abs(interpolated_fft)
                total_chi_sq += chi_sq(corr_flux[str(wavelength)],
                                       corr_flux_err[str(wavelength)],
                                       corr_flux_model)\
                    * OPTIONS["fit.chi2.weight.corr_flux"]
                # cphase_model = ...
                # total_chi_sq += chi_sq(cphase, cphase_err, cphase_model)\
                    # * OPTIONS["fit.chi2.weight.cphases"]
            else:
                continue
    return np.array(total_chi_sq)


def initiate_randomly(free_parameters: Dict[str, Parameter],
                      nwalkers: float) -> np.ndarray:
    """initialises a random numpy.ndarray from the parameter's limits.

    parameters
    -----------
    free_parameters : dict of Parameter

    Returns
    -------
    numpy.ndarray
    """
    initial = np.ndarray([nwalkers, len(free_parameters)])
    for index, parameter in enumerate(free_parameters.values()):
        initial[:, index] = np.random.random(nwalkers)\
            * np.ptp([parameter.min, parameter.max])+parameter.min
    return initial


def run_mcmc(nwalkers: int,
             nsteps: int,
             discard: int,
             save_path: Optional[Path] = None) -> np.array:
    """Runs the emcee Hastings Metropolitan sampler

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    The burn-in is first run to explore the parameter space and then the
    walkers settle into the maximum of the density. The state variable is
    then passed to the production run.

    The chain is reset before the production with the state variable being
    passed. 'rstate0' is the state of the internal random number generator

    Parameters
    ----------
    parameters : dict of Parameter
    nwalkers : int
    nsteps : int
    discard : int
    wavelengths : float
    save_path: pathlib.Path, optional

    Returns
    -------
    np.ndarray
    """
    initial = initiate_randomly(OPTIONS["model.params"], nwalkers)
    with Pool() as pool:
        print(f"Executing MCMC with {cpu_count()} cores.")
        print("--------------------------------------------------------------")
        sampler = emcee.EnsembleSampler(nwalkers, len(OPTIONS["model.params"]),
                                        lnprob, pool=pool)
        sampler.run_mcmc(initial, nsteps, progress=True)
    return (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]


if __name__ == "__main__":
    # OPTIONS["fourier.binning"] = 1
    # star = Star(dim=DIM, dist=DISTANCE, eff_temp=EFF_TEMP, lum=LUMINOSITY)
    # temp_grad = AsymmetricSDGreyBodyContinuum(dim=DIM, dist=DISTANCE, Tin=INNER_TEMP,
    #                                           pixel_size=PIXEL_SIZE, Teff=EFF_TEMP,
    #                                           kappa_abs=KAPPA_ABS, lum=LUMINOSITY,
    #                                           kappa_cont=KAPPA_ABS_CONT,
    #                                           rin=0.5, rout=100, Mdust=0.11,
    #                                           q=0.5, p=0.5, a=0.5, cont_weight=0.5,
    #                                           pa=150, phi=33, elong=0.5)
    # model = Model([star, temp_grad])
    # model.plot_model(4096, 0.1, 4.78301581e-06)
    # plt.show()
    # run_mcmc(nwalkers=25, nsteps=100, discard=10)
    initial = initiate_randomly(PARAMS, 25)
    lnprob(initial[0])
