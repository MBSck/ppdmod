#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect
import time

from typing import Any, Dict, List, Union, Optional, Callable
from astropy.io import fits
from functools import wraps

from .functionality.constants import *

# TODO: Make progress bar into a decorator and also keep the time of the
# process and show the max time

# TODO: Finish the fit function, but maybe implement it in the plotter instead?

# Functions

def progress_bar(progress: int, total: int):
    """Displays a progress bar

    Parameters
    ----------
    progress: int
        Total progress
    total: int
        Total iterations
    """
    percent = 100 * (progress/total)
    bar = '#' * int(percent) + '-' * (100-int(percent))
    print(f"\r|{bar}|{percent:.2f}% - {progress}/{total}", end='\r')

def trunc(values, decs=0):
    """Truncates the floating point decimals"""
    return np.trunc(values*10**decs)/(10**decs)

def chi_sq(data: np.ndarray, sigma_sq: np.ndarray,
           model: np.ndarray) -> float:
    """The chi square minimisation"""
    return np.sum(np.log(2*np.pi*sigma_sq) + (data-model)**2/sigma_sq)

def get_rndarr_from_bounds(bounds: List,
                           centre_rnd: Optional[bool] = False):
    """Initialises a random float/list via a normal distribution from the
    bounds provided

    Parameters
    -----------
    bounds: List
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]
    centre_rnd: bool, optional
        Get a random number close to the centre of the bound

    Returns
    -------
    float | np.ndarray
    """
    initial = []

    if centre_rnd:
        for lower, upper in bounds:
            if upper == 2:
                initial.append(np.random.normal(1.5, 0.2))
            else:
                initial.append(np.random.normal(upper/2, 0.2))

    else:
        for lower, upper in bounds:
            initial.append(np.random.uniform(lower, upper))

    return initial

def plot_amp_phase_comparison(amp_data: List, cphase_data: List,
                              baselines: List, t3phi_baselines: List,
                              matplot_axes: Optional[List] = []) -> None:
    """Plots the deviation of a model from real data of an object for both
    amplitudes and phases (closure phases)

    Parameters
    ----------
    amp_data: List
        Contains both the model's and the real object's amplitude data and
        errors in the following format [[real_obj, real_err], [model]
    cphase_data: List
        Contains both the model's and the real object's closure phase data and
        errors in the following format [[real_obj, real_err], [model]]
    baselines: List
        The baselines of the amplitudes
    t3phi_baselines: List
        The baselines of the closure phases
    matplot_axes: List, optional
        The axes of matplotlib if this plot is to be embedded in an already
        existing one
    """
    if matplot_axes:
        ax, bx = matplot_axes
    else:
        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
        ax, bx = axarr.flatten()

    amp, amperr = amp_data[0]
    amp_mod = amp_data[1][0]
    cphase, cphaseerr = cphase_data[0]
    cphase_mod = cphase_data[1][0]

    all_amp = np.concatenate([amp, amp_mod])
    y_min_amp, y_max_amp = 0, np.max(all_amp)
    y_space_amp = np.sqrt(y_max_amp**2+y_min_amp**2)*0.1
    y_lim_amp = [y_min_amp-y_space_amp, y_max_amp+y_space_amp]

    all_cphase = np.concatenate([cphase, cphase_mod])
    y_min_cphase, y_max_cphase = np.min(all_cphase), np.max(all_cphase)
    y_space_cphase = np.sqrt(y_max_cphase**2+y_min_cphase**2)*0.1
    y_lim_cphase = [y_min_cphase-y_space_cphase, y_max_cphase+y_space_cphase]

    ax.errorbar(baselines, amp, amperr,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    ax.scatter(baselines, amp_mod, marker='X', label="Model data")
    ax.set_xlabel("Baselines [m]")
    ax.set_ylabel("Correlated fluxes [Jy]")
    ax.set_ylim(y_lim_amp)
    ax.legend(loc="upper right")

    bx.errorbar(t3phi_baselines, cphase, cphaseerr,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    bx.scatter(t3phi_baselines, cphase_mod, marker='X', label="Model data")
    bx.set_xlabel("Longest baselines [m]")
    bx.set_ylabel(fr"Closure Phases [$^\circ$]")
    bx.set_ylim(y_lim_cphase)
    bx.legend(loc="upper right")

def plot_txt(ax, title_dict: Dict, text_dict: Dict,
             text_font_size: Optional[int] = 12) -> None:
    """Makes a plot with only text information

    Parameters
    ----------
    ax
        The axis of matplotlib
    input_dict: Dict
        A dict that contains the text as a key and the info as the value
    """
    plot_title = "\n".join([r"$\mathrm{%s}$" % (i) if o == ""\
                            else r"$\mathrm{%s}$: %.2f" % (i.lower(), o)\
                            for i, o in title_dict.items()])
    ax.annotate(plot_title, xy=(0, 1), xytext=(12, -12), va='top',
        xycoords='axes fraction', textcoords='offset points', fontsize=16)
    ax.set_title(plot_title)

    text = "\n".join([r"$\mathrm{%s}$" % (i) if o == ""\
                            else r"$\mathrm{%s}$: %.2f" % (i, o)\
                      for i, o in text_dict.items()])
    ax.annotate(text, xy=(0, 0), xytext=(12, -12), va="bottom",
                xycoords='axes fraction', textcoords='offset points',
                fontsize=text_font_size)

    plt.tight_layout()
    ax.axis('off')

def rotation_synthesis_uv(inp):
    """This function was written by Jozsef Varga (from menEWS: menEWS_plot.py).

    Calculates uv-point corresponding to inp (see "get_header_info"),
    for hour angle(s) (ha)
    """
    ra, dec, BE, BN, BL, base = inp
    paranal_lat = -24.62587 * np.pi / 180.

    u = BE * np.cos(ha) -\
            BN * np.sin(lat) * np.sin(ha) + BL * np.cos(lat) * np.sin(ha)
    v = BE * np.sin(dec) * np.sin(ha) +\
            BN * (np.sin(lat) * np.sin(dec) * np.cos(ha) +\
                  np.cos(lat) * np.cos(dec)) - BL * \
        (np.cos(lat) * np.sin(dec) * np.cos(ha)- np.sin(lat) * np.cos(dec))
    return u, v

def make_uv_tracks(uv, inp, flag, ax, bases=[], symbol='x',color='',
    print_station_names=True,sel_wl=1.0,plot_Mlambda=False):
    """This function was written by Jozsef Varga (from menEWS: menEWS_plot.py).

    From coordinate + ha (range), calculate uv tracks"""

    ra, dec, BE, BN, BL, base = inp
    paranal_lat = -24.62587 * np.pi / 180.
    mlim = 2.0  # airmass limit for tracks

    if plot_Mlambda == True:
        u, v = map(lambda x: x/sel_wl, uv)
    else:
        u, v = uv

    if not color:
        if np.all(flag) == 'True':
            color = 'r'
        else:
            color = 'g'

    if base not in bases:
        hamax = np.arccos(abs((1. / mlim - np.sin(lat) * np.sin(dec)) / \
                              (np.cos(lat) * np.cos(dec))))
        harng = np.linspace(-hamax, hamax, 1000)

        ul, vl = ulvl = calculate_uv_points(inp, harng)
        if plot_Mlambda == True:
            u, v = map(lambda x: x/sel_wl, ulvl)

        ax.plot(ul, vl, '-', color='grey',alpha=0.5)
        ax.plot(-ul, -vl, '-', color='grey',alpha=0.5)
        ax.plot([0.], [0.], '+k', markersize=5, markeredgewidth=2,alpha=0.5)

        if print_station_names:
            ax.text(-u-7, -v-3, base, color='0',alpha=0.8)
        bases.append(base)

    ax.plot(u, v, symbol, color=color, markersize=10, markeredgewidth=3)
    ax.plot(-u, -v, symbol, color=color, markersize=10, markeredgewidth=3)

    return bases

def make_uv_plot(dic,ax,verbose=False,annotate=True,B_lim=(np.nan,np.nan),figsize=(5,5),
    color='',print_station_names=True,sel_wl=1.0,plot_Mlambda=False):
    """This function was written by Jozsef Varga (from menEWS: menEWS_plot.py)"""
    if plot_Mlambda==False:
        sel_wl = 1.0
    try:
        u = dic['VIS2']['U']
        v = dic['VIS2']['V']
        flag = dic['VIS2']['FLAG']
        sta_index = dic['VIS2']['STA_INDEX']
        mjd = dic['VIS2']['MJD']
    except KeyError as e:
        if verbose: print(e)
        u = [0.0]
        v = [0.0]
        flags = [False]
        sta_index = []
        mjd = [0.0]

    uvs = []
    inps = []
    flags = []
    umax = []
    vmax = []
    for j in range(len(u)):
        uvs.append([u[j],v[j]])
        try:
            BE, BN, BL = dic['STAXYZ'][sta_index[j, 0] == dic['STA_INDEX']][0] - \
                dic['STAXYZ'][sta_index[j, 1] == dic['STA_INDEX']][0]
            sta_label= dic['STA_NAME'][sta_index[j, 0] == dic['STA_INDEX']][0] + '-' + \
                        dic['STA_NAME'][sta_index[j, 1] == dic['STA_INDEX']][0]
        except IndexError as e:
            print('make_uv_plot STA_INDEX error.')
            print(e)
            BE, BN, BL = [np.nan,np.nan,np.nan]
            sta_label= ''
        inps.append( [dic['RA'] * np.pi / 180., dic['DEC'] * np.pi / 180., BE, BN, BL, sta_label]  )
        flags.append(flag[j])
    bases = []
    umax = np.nanmax(np.abs(u))
    vmax = np.nanmax(np.abs(v))
    if not (dic['MJD-OBS']):
        dic['MJD-OBS'] = np.amin(mjd[0])
    try:
        rel_time = (mjd - dic['MJD-OBS']) * 24.0 * 3600.0  # (s)
        dic['TREL'] = rel_time[0]

        for k, uv in enumerate(uvs):
            bases = make_uv_tracks(uv, inps[k], flags[k],ax, bases,
            color=color,print_station_names=print_station_names,
            sel_wl=sel_wl,plot_Mlambda=plot_Mlambda)

        if plot_Mlambda == False:
            xlabel ='$u$ (m)'
            ylabel ='$v$ (m)'
        else:
            xlabel ='$u$ ($M\lambda$)'
            ylabel ='$v$ ($M\lambda$)'
        ax.set_xlim((130, -130))
        ax.set_ylim((-130, 130))
        plotmax = 1.3*np.amax([umax,vmax])

        plot_title = dic['TARGET'] + "\n" + "date: " + dic['DATE-OBS'] + "\n" + "TPL start: " + dic['TPL_START'] + "\n" + dic['CATEGORY'] + ' ' +\
            dic['BAND'] + ' ' + dic['DISPNAME'] #+ ' ' + dic['BCD1'] + '-' + dic['BCD2']
        if math.isnan(B_lim[0]):
            xlim = (+plotmax/ sel_wl,-plotmax/ sel_wl)
            ylim = (-plotmax/ sel_wl,+plotmax/ sel_wl)
        else:
            xlim = (+B_lim[1]/ sel_wl,-B_lim[1]/ sel_wl)
            ylim = (-B_lim[1]/ sel_wl,+B_lim[1]/ sel_wl)
        #if plot_Mlambda == True:
        plot_config(xlabel, ylabel,plot_title, ax, dic,
                    ylim=ylim,xlim=xlim,plot_legend=False,annotate=annotate)
    except TypeError as e:
        if verbose: print('Unable to plot ' + 'uv')
        if verbose: print(e)
        return 1

    return 0

def timeit(func):
    """Simple timer decorator for functions"""
    @wraps(func)
    def timed_func(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        et = time.time()
        print(f"{func.__name__} execution took: {et-st} sec")
        return result
    return timed_func

def orbit_au2arc(orbit_radius: Union[int, float],
                 distance: Union[int, float]):
    """Converts the orbital radius from [au] to [arc]

    Parameters
    ----------
    orbit_radius: int | float
        The radius of the star or its orbit
    distance: int | float
        The distance to the star

    Returns
    -------
    orbit: float
        The orbit in arcseconds
    """
    return orbit_radius/distance

def m2au(radius: Union[int, float]):
    """Converts units of [m] to [au]"""
    return radius/AU_M

def m2arc(radius: float, distance: int):
    """Converts [m] to [arcsec]"""
    return orbit_au2arc(m2au(radius), distance)

def m2mas(radius: float, distance: int):
    """Converts [m] to [mas]"""
    return m2arc(radius, distance)*1000

def m2rad(radius: float, distance: int):
    """Converts [m] to [rad]"""
    return arc2rad(m2arc(radius, distance))

def arc2rad(length_in_arc: Union[int, float]):
    """Converts the orbital radius from [arcsec] to [rad]"""
    return length_in_arc*ARCSEC2RADIANS

def mas2rad(angle: Optional[Union[int, float, np.ndarray]] = None):
    """Returns a given angle in mas/rad or the pertaining scaling factor

    Parameters
    ----------
    angle: int | float | np.ndarray, optional
        The input angle(s)

    Returns
    -------
    float
        The angle in radians
    """
    if angle is None:
        return np.radians(1/3.6e6)
    return np.radians(angle/3.6e6)

def rad2mas(angle: Optional[float] = None):
    """Converts from radians to milliarcseconds"""
    return 1/mas2rad(angle)

def sr2mas(mas_size: float, sampling: int):
    """Converts the dimensions of an object from 'sr' to 'mas'. the result is
    in per pixel

    Parameters
    ----------
    mas_size: float
        The size of the image [mas]
    sampling: int
        The pixel sampling of the image
    """
    return (mas_size/(sampling*3600e3*180/np.pi))**2

def azimuthal_modulation(image, polar_angle: Union[float, np.ndarray],
                         modulation_angle: float,
                         amplitude: int  = 1) -> Union[float, np.ndarray]:
    """Azimuthal modulation of an object

    Parameters
    ----------
    polar_angle: float | np.ndarray
        The polar angle of the x, y-coordinates
    amplitude: int
        The 'c'-amplitude

    Returns
    -------
    azimuthal_modulation: float | np.ndarray
    """
    # TODO: Implement Modulation field like Jozsef?
    modulation_angle = np.radians(modulation_angle)
    total_mod = (amplitude*np.cos(polar_angle-modulation_angle))

    image *= np.array(1 + total_mod)
    image[image < 0.0] = 0.0

    return image

def set_grid(mas_size: int, size: int, sampling: Optional[int] = None,
             incline_params: Optional[List[float]] = None) -> np.array:
    """Sets the size of the model and its centre. Returns the polar coordinates

    Parameters
    ----------
    mas_size: int
        Sets the size of the image [mas]
    size: int
        Sets the range of the model image and implicitly the x-, y-axis.
        Size change for simple models functions like zero-padding
    sampling: int, optional
        The pixel sampling
    incline_params: List[float], optional
        A list of the inclination parameters [axis_ratio, pos_angle] [mas, rad]

    Returns
    -------
    radius: np.array
        The radius
    xc: np.ndarray
        The x-axis used to calculate the radius
    """
    fov_scale = mas_size/size

    if sampling is None:
        sampling = size

    x = np.linspace(-size//2, size//2, sampling, endpoint=False)*fov_scale
    y = x[:, np.newaxis]

    if incline_params:
        try:
            axis_ratio, pos_angle = incline_params
        except:
            raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                          " arguments, ellipsis_angles must be of the"
                          " form [axis_ratio, pos_angle]")

        if axis_ratio < 1.:
            raise ValueError("The axis_ratio has to be bigger than 1.")

        if (pos_angle < 0) or (pos_angle > 180):
            raise ValueError("The positional angle must be between [0, 180]")

        pos_angle = np.radians(pos_angle)

        xr, yr = x*np.cos(pos_angle)+y*np.sin(pos_angle),\
                (-x*np.sin(pos_angle)+y*np.cos(pos_angle))/axis_ratio
        radius = np.sqrt(xr**2+yr**2)
        axis, phi = [xr, yr], np.arctan2(xr, yr)
    else:
        radius = np.sqrt(x**2+y**2)
        axis, phi = [x, y], np.arctan2(x, y)

    return radius, axis, phi

def zoom_array(array: np.ndarray, bounds: List) -> np.ndarray :
    """Zooms in on an image by cutting of the zero-padding

    Parameters
    ----------
    array: np.ndarray
        The image to be zoomed in on
    bounds: int
        The boundaries for the zoom, the minimum and maximum

    Returns
    -------
    np.ndarray
        The zoomed in array
    """
    min_ind, max_ind = bounds
    return array[min_ind:max_ind, min_ind:max_ind]

def set_uvcoords(wavelength: float, sampling: int, size: Optional[int] = 200,
                 angles: List[float] = None, uvcoords: np.ndarray = None,
                 B: Optional[bool] = True) -> np.array:
    """Sets the uv coords for visibility modelling

    Parameters
    ----------
    wavelength: float
        The wavelength the (u,v)-plane is sampled at
    sampling: int
        The pixel sampling
    size: int, optional
        Sets the range of the (u,v)-plane in meters, with size being the
        longest baseline
    angles: List[float], optional
        A list of the three angles [ellipsis_angle, pos_angle inc_angle]
    uvcoords: List[float], optional
        If uv-coords are given, then the visibilities are calculated for
    B: bool, optional
        Returns the baseline vector if toggled true, else the r vector

    Returns
    -------
    baselines: ArrayLike
        The baselines for the uvcoords
    uvcoords: ArrayLike
        The axis used to calculate the baselines
    """
    if uvcoords is None:
        axis = np.linspace(-size, size, sampling, endpoint=False)

        # Star overhead sin(theta_0)=1 position
        u, v = axis/wavelength, axis[:, np.newaxis]/wavelength

    else:
        axis = uvcoords/wavelength
        u, v = np.array([i[0] for i in uvcoords]), \
                np.array([i[1] for i in uvcoords])

    if angles is not None:
        try:
            if len(angles) == 1:
                pos_angle = np.radians(angles[0])
                ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                        v*np.cos(pos_angle)-u*np.sin(pos_angle)
                r = np.sqrt(ur**2+vr**2)
                B_vec = r*wavelength if B else r
            else:
                axis_ratio = angles[0]
                pos_angle, inc_angle = map(lambda x: np.radians(x), angles[1:])

                ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                        (v*np.cos(pos_angle)-u*np.sin(pos_angle))/axis_ratio
                r = np.sqrt(ur**2+vr**2*np.cos(inc_angle)**2)
                B_vec = r*wavelength if B else r

            axis = [ur, vr]
        except:
            raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                          " arguments, ellipsis_angles must be of the form"
                          " either [pos_angle] or "
                          " [ellipsis_angle, pos_angle, inc_angle]")

    else:
        r = np.sqrt(u**2+v**2)
        B_vec = r*wavelength if B else r
        axis = [u, v]

    return B_vec, axis

def stellar_radius_pc(T_eff: int, L_star: int):
    """Calculates the stellar radius from its attributes and converts it from
    m to parsec

    Parameters
    ----------
    T_eff: int
        The star's effective temperature [K]
    L_star: int
        The star's luminosity [L_sun]

    Returns
    -------
    stellar_radius: float
        The star's radius [pc]
    """
    L_star *= SOLAR_LUMINOSITY
    stellar_radius_m = np.sqrt(L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*T_eff**4))
    return stellar_radius_m/PARSEC2M

def sublimation_temperature(r_sub: float, L_star: int, distance: int):
    """Calculates the sublimation temperature at the inner rim of the disk

    Parameters
    ----------
    r_sub: float
        The sublimation radius [mas]
    L_star: int
        The star's luminosity in units of nominal solar luminosity
    distance: int
        Distance in parsec

    Returns
    -------
    T_sub: float
        The sublimation temperature [K]
    """
    L_star *= SOLAR_LUMINOSITY
    r_sub /= m2mas(1, distance)
    return (L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*r_sub**2))**(1/4)

def sublimation_radius(T_sub: int, L_star: int, distance: int):
    """Calculates the sublimation radius of the disk

    Parameters
    ----------
    T_sub: int
        The sublimation temperature of the disk. Usually fixed to 1500 K
    L_star: int
        The star's luminosity in units of nominal solar luminosity
    distance: int
        Distance in parsec

    Returns
    -------
    R_sub: int
        The sublimation_radius [mas]
    """
    L_star *= SOLAR_LUMINOSITY
    sub_radius_m = np.sqrt(L_star/(4*np.pi*STEFAN_BOLTZMAN_CONST*T_sub**4))
    return m2mas(sub_radius_m, distance)

def temperature_gradient(radius: float, r_0: Union[int, float],
                         q: float, T_0: int) -> Union[float, np.ndarray]:
    """Temperature gradient model determined by power-law distribution.

    Parameters
    ----------
    radius: float
        The specified radius
    r_0: float
        The initial radius
    q: float
        The power-law index
    T_0: float
        The temperature at r_0

    Returns
    -------
    temperature: float | np.ndarray
        The temperature at a certain radius
    """
    # q is 0.5 for flared irradiated disks and 0.75 for standard viscuous disks
    with np.errstate(divide='ignore'):
        return T_0*(radius/r_0)**(-q)

def plancks_law_nu(T: Union[float, np.ndarray],
                   wavelength: float) -> [float, np.ndarray]:
    """Gets the blackbody spectrum at a certain T(r). Wavelength and
    dependent. The wavelength will be converted to frequency

    Parameters
    ----------
    T: float
        The temperature of the blackbody
    wavelength: float
        The wavelength to be converted to frequency

    Returns
    -------
    planck's law/B_nu(nu, T): float | np.ndarray
        The spectral radiance (the power per unit solid angle) of a black-body
        in terms of frequency
    """
    nu = SPEED_OF_LIGHT/wavelength
    factor = (2*PLANCK_CONST*nu**3)/SPEED_OF_LIGHT**2
    exponent = (PLANCK_CONST*nu)/(BOLTZMAN_CONST*T)

    return factor*(1/(np.exp(exponent)-1))

def do_fit():
    """Does automatic gauss fits"""
    # Fits the data
    scaling_rad2arc = 206265

    # Gaussian fit
    fwhm = 1/scaling_rad2arc/1000           # radians

    # np.linspace(np.min(spat_freq), np.max(spat_freq), 25)
    xvals, yvals = self.baseline_distances, self.mean_bin_vis2
    pars, cov = curve_fit(f=gaussian, xdata=xvals, ydata=yvals, p0=[fwhm], bounds=(-np.inf, np.inf))
    # xvals = np.linspace(50, 150)/3.6e-6
    # fitted_model= np.square(gaussian(xvals, fwhm))
    ax.plot(xvals, gaussian(xvals, pars), label='Gaussian %.1f"'%(fwhm*scaling_rad2arc*1000))

    # Airy-disk fit
    fwhm = 3/scaling_rad2arc/1000           # radians
    fitted_model = np.square(airy(xvals, fwhm))
    ax.plot(xvals/1e6, fitted_model*0.15, label='Airy Disk %.1f"'%(fwhm*scaling_rad2arc*1000))
    ax.set_ylim([0, 0.175])
    ax.legend(loc='best')


if __name__ == "__main__":
    ...

