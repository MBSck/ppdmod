from astropy.units.format import base
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.units import Quantity
from typing import Dict, List, Optional

from .data_prep import DataHandler


def plot(image: Quantity) -> None:
    """Plots and image"""
    plt.imshow(image.value)
    plt.show()

def print_results(best_fit_total_fluxes, best_fit_corr_fluxes,
                  best_fit_cphases, data: DataHandler) -> None:
    """Prints the model's values"""
    print("Best fit total fluxes:")
    print(best_fit_total_fluxes)
    print("Best real total fluxes:")
    print(data.total_fluxes)
    print("--------------------------------------------------------------")
    print("Best fit correlated fluxes:")
    print(best_fit_corr_fluxes)
    print("Real correlated fluxes:")
    print(data.corr_fluxes)
    print("--------------------------------------------------------------")
    print("Best fit cphase:")
    print(best_fit_cphases)
    print("Real cphase:")
    print(data.cphases)
    print("--------------------------------------------------------------")
    print("Theta max:")
    print(data.theta_max)

def plot_fit_results(best_fit_total_fluxes, best_fit_corr_fluxes,
                     best_fit_cphases, data: DataHandler,
                     plot_px_size: Optional[int] = 1024,
                     save_path: Optional[Path] = None) -> None:
    """Plot the samples to get estimate of the density that has been sampled,
    to test if sampling went well

    Parameters
    ----------
    """
    print_results(best_fit_total_fluxes, best_fit_corr_fluxes, best_fit_cphases, data)
    plot_wl = data.wavelengths[0]
    fig, axarr = plt.subplots(2, 3, figsize=(20, 10))
    ax, bx, cx = axarr[0].flatten()
    ax2, bx2, cx2 = axarr[1].flatten()

    # title_dict = {"Model Fit Parameters": ""}
    # text_dict = { "FOV": pixel_size, "npx": sampling,
                 # "zero pad order": zero_padding_order, "wavelength": plot_wl,
                 # "": "", "blackbody params": "", "---------------------": "",
                 # **bb_params_dict, "": "", "best fit values": "",
                 # "---------------------": "", **theta_max_dict, "": "",
                 # "hyperparams": "", "---------------------": "",
                 # **hyperparams_dict}

    # plot_txt(ax, title_dict, text_dict, text_font_size=10)
    plot_amp_phase_comparison(data, best_fit_total_fluxes,
                              best_fit_corr_fluxes, best_fit_cphases)
    data.fourier.plot_amp_phase(data, best_fit_total_fluxes,
                                best_fit_corr_fluxes, best_fit_cphases,
                                matplot_axes=[fig, ax2, bx2, cx2])
    plot_name = f"Best-fit-model for {(plot_wl*1e6):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))
    plt.tight_layout()
    plt.show()


def plot_amp_phase_comparison(data: DataHandler, best_fit_total_fluxes,
                              best_fit_corr_fluxes, best_fit_cphases,
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

    all_amp = np.concatenate((data.corr_fluxes.value, best_fit_corr_fluxes))
    y_min_amp, y_max_amp = 0, np.max(all_amp)
    y_space_amp = np.sqrt(y_max_amp**2+y_min_amp**2)*0.1
    y_lim_amp = [y_min_amp-y_space_amp, y_max_amp+y_space_amp]

    all_cphase = np.concatenate((data.cphases.value, best_fit_cphases))
    y_min_cphase, y_max_cphase = np.min(all_cphase), np.max(all_cphase)
    y_space_cphase = np.sqrt(y_max_cphase**2+y_min_cphase**2)*0.1
    y_lim_cphase = [y_min_cphase-y_space_cphase, y_max_cphase+y_space_cphase]

    # TODO: Make this into a quantity immedieatly?
    baselines = [baseline.value for baseline in data.baselines]

    ax.errorbar(baselines, data.corr_fluxes.value, data.corr_fluxes_error.value,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    ax.scatter(baselines.value, best_fit_corr_fluxes, marker='X', label="Model data")
    ax.set_xlabel("Baselines [m]")
    ax.set_ylabel("Correlated fluxes [Jy]")
    ax.set_ylim(y_lim_amp)
    ax.legend(loc="upper right")

    longest_baselines = np.array([np.sort(baselines.value.flatten())[::-1][:4]\
                                  for baselines in data.uvcoords_cphase])
    bx.errorbar(longest_baselines, data.cphases.value, data.cphases_error.value,
                color="goldenrod", fmt='o', label="Observed data", alpha=0.6)
    bx.scatter(longest_baselines, best_fit_cphases, marker='X', label="Model data")
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


if __name__ == "__main__":
    ...

