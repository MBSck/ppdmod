from typing import Union, Optional, Tuple, Dict, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from component import NumericalComponent
from parameter import Parameter
from utils import rebin_image


class Model:
    """The oimModel class hold a model made of one or more components (derived
    from the oimComponent class).

    It allows to compute images (or image cubes for wavelength or time
    dependent models) and complex coherent fluxes for a vector of u,v,wl,
    and t coordinates.

    Parameters
    ----------
    *components : list of oimComponent
       The components of the model.

    Attributes
    ----------
    components : list of oimComponent
       The components of the model.
    """

    def __init__(self, *components: List[NumericalComponent]) -> None:
        """Constructor of the class"""
        if len(components) == 1 and type(components[0]) == list:
            self.components = components[0]
        else:
            self.components = components

    @property
    def params(self) -> Dict[str, Parameter]:
        """Get the Model parameters.

        Parameters
        ----------
        free : bool, optional
            If True retrieve the free parameters of the models only.
            The default is False.

        Returns
        -------
        params : dict of Parameter
            Dictionary of the model's parameters.
        """
        params = {}
        for index, component in enumerate(self.components):
            for name, parameter in component.params.items():
                if parameter not in params.values():
                    params[f"c{index+1}_{component.shortname.replace(' ', '_')}_{name}"] = parameter
        return params

    @property
    def free_params(self) -> Dict[str, Parameter]:
        """Get the Model free paramters

        Returns
        -------
        dict of Parameter
            A Dictionary of the model's free parameters.
        """
        return {key: value for key, value in self.params if value.free}

    def calculate_complex_visibility(self, ucoord: ArrayLike, vcoord: ArrayLike,
                                     wl: Optional[ArrayLike] = None) -> np.ndarray:
        """Compute and return the complex coherent flux for an array of u,v
        (and optionally wavelength and time) coordinates.

        Parameters
        ----------
        ucoord : array_like
            Spatial coordinate u (in cycles/rad).
        vcoord : array_like
            Spatial coordinate v (in cycles/rad).
        wl : array_like, optional
            Wavelength(s) in meter. The default is None.

        Returns
        -------
        numpy.ndarray
            The complex coherent flux. The same size as u & v.
        """
        res = complex(0, 0)
        for component in self.components:
            res += component.getComplexCoherentFlux(ucoord, vcoord, wl)
        return res

    def calculate_image(self, dim: int, pixSize: float,
                        wl: Optional[Union[float, ArrayLike]] = None,
                        fromFT: Optional[bool] = False) -> np.ndarray:
        """Compute and return an image.

        The returned image as the x,y dimension dim in pixel with
        an angular pixel size pixSize in rad. Image is returned as a numpy
        array unless the keyword fits is set to True. In that case the image is
        returned as an astropy.io.fits hdu.

        Parameters
        ----------
        dim : int
            Image x & y dimension in pixels.
        pixSize : float
            Pixel angular size in mas.
        wl : int or array_like, optional
            Wavelength(s) in meter. The default is None.
        fromFT : bool, optional
            If True compute the image using FT formula when available.
            The default is False.

        Returns
        -------
        numpy.ndarray
             A numpy 2D array (or 3D array if wl is given).
        """
        if fromFT:
            v = np.linspace(-0.5, 0.5, dim)
            vx, vy = np.meshgrid(v, v)
            spfx_arr, spfy_arr = (vx/pixSize/u.mas.to(u.rad)).flatten()
            spfy_arr = (vy/pixSize/u.mas.to(u.rad)).flatten()

            ft = self.calculate_complex_visibility(spfx_arr, spfy_arr, wl_arr)
            image = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft))))
        else:
            image = None
            for component in self.components:
                if image is None:
                    image = component.calculate_image(dim, pixSize, wl)
                else:
                    image += component.calculate_image(dim, pixSize, wl)
        return image

    def plot_model(self, dim: int, pixSize: float,
                   wl: Optional[Union[int, ArrayLike]] = None,
                   fromFT: Optional[bool] = False,
                   axe: Optional[Axes] = None,
                   figsize: Optional[Tuple[float]] = (3.5, 2.5),
                   savefig: Optional[str] = None,
                   colorbar: Optional[bool] = True,
                   legend: Optional[bool] = False,
                   kwargs_legend: Dict = {},
                   binning_factor: Optional[bool] = None,
                   **kwargs: Dict) -> Tuple[Figure, Axes, np.ndarray]:
        """Show the mode Image or image-Cube

        Parameters
        ----------
        dim : int
            Image x & y dimension in pixels.
        pixSize : float
            Pixel angular size in mas.
        wl : int or array_like, optional
            Wavelength(s) in meter. The default is None.
        fromFT : bool, optional
            If True compute the image using FT formula when available.
            The default is False.
        axe : matplotlib.axes.Axes, optional
            If provided the image will be shown in this axe. If not a new figure
            will be created. The default is None.
        figsize : tuple of float, optional
            The Figure size in inches. The default is (8., 6.).
        savefig : str, optional
            Name of the files for saving the figure If None the figure is not saved.
            The default is None.
        colorbar: bool, optional
            Add a colobar to the Axe. The default is True.
        legend : bool, optional
            If True displays a legend. Default is False.
        kwargs_legend: dict, optional
        rebin : bool, optional
            If True rebin the image according to OPTIONS["FTBinningFactor"].
        kwargs : dict
            Arguments to be passed to the plt.imshow function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure created if needed.
        axe : matplotlib.axes.Axes
            The Axes instances, created if needed.
        im  : numpy.array
            The image(s).
        """
        image = self.calculate_image(dim, pixSize, wl, fromFT=fromFT)
        if axe is None:
            fig, axe = plt.subplots(figsize=figsize)
        else:
            try:
                fig = axe.get_figure()
            except Exception:
                fig = axe.flatten()[0].get_figure()

        cb = axe.imshow(image[:],
                        vmax=np.sort(image[:].flatten())[::-1][1],
                        extent=[-dim/2*pixSize, dim/2*pixSize,
                                -dim/2*pixSize, dim/2*pixSize],
                        origin='lower', **kwargs)

        axe.set_xlim(dim/2*pixSize, -dim/2*pixSize)
        axe.set_xlabel("$\\alpha$(mas)")
        axe.set_ylabel("$\\delta$(mas)")

        if legend:
            txt = ""
            if wl is not None:
                txt += r"wl={:.4f}$\mu$m\n".format(wl*1e6)
            if "color" not in kwargs_legend:
                kwargs_legend["color"] = "w"
            axe.text(0, 0.95*dim/2*pixSize, txt,
                     va="top", ha='center', **kwargs_legend)

        if colorbar:
            fig.colorbar(cb, ax=axe, label="Normalized Intensity")

        if savefig is not None:
            plt.savefig(savefig)
        return fig, axe, image

    def plot_fourier(self, dim: int, pixSize: float, wl: float,
                     axe: Optional[Axes] = None,
                     figsize: Optional[Tuple[float]] = (3.5, 2.5),
                     savefig: Optional[str] = None,
                     colorbar: Optional[bool] = True,
                     legend: Optional[bool] = False,
                     display_mode: Optional[str] = "corr_flux",
                     kwargs_legend: Optional[Dict] = {},
                     **kwargs: Dict):
        """Show the amplitude and phase of the Fourier space

        Parameters
        ----------
        dim : int
            Image x & y dimension in pixels.
        pixSize : float
            Pixel angular size in mas.
        wl : int or array_like, optional
            Wavelength(s) in meter. The default is None.
        t :  int or array_like, optional
            Time in s (mjd). The default is None.
        axe : matplotlib.axes.Axes, optional
            If provided the image will be shown in this axe. If not a new figure
            will be created. The default is None.
        normPow : float, optional
            Exponent for the Image colorscale powerLaw normalisation.
            The default is 0.5.
        figsize : tuple of float, optional
            The Figure size in inches. The default is (8., 6.).
        savefig : str, optional
            Name of the files for saving the figure If None the figure is not saved.
            The default is None.
        colorbar : bool, optional
            Add a colobar to the Axe. The default is True.
        legend : bool, optional
            If True displays a legend. Default is False.
        swapAxes : bool, optional
            If True swaps the axes of the wavelength and time.
            Default is True.
        display_mode : str, optional
            Displays either the amplitude "amp" or the phase "phase".
            Default is "amp".
        kwargs_legend: dict, optional
        normalize : bool, optional
            If True normalizes the image.
        **kwargs : dict
            Arguments to be passed to the plt.imshow function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure created if needed
        axe : matplotlib.axes.Axes
            The Axes instances, created if needed.
        im  : numpy.ndarray
            The image(s).
        """
        v = np.linspace(-0.5, 0.5, dim)
        vx, vy = np.meshgrid(v, v)
        spfx_arr, spfy_arr = map(lambda x: (x/pixSize/u.mas.to(u.rad)).flatten(),
                                 [vx, vy])
        spfx_extent = spfx_arr.max()
        fourier_transform = self.calculate_complex_visibility(spfx_arr, spfy_arr)
        if display_mode == "vis":
            im = np.abs(fourier_transform)
            im /= im.max()
        elif display_mode == "corr_flux":
            im = np.abs(fourier_transform)
        elif display_mode == "phase":
            im = np.angle(fourier_transform, deg=True)
        else:
            raise NameError("Only 'vis', 'corr_flux' and 'phase' are valid"
                            " choices for the display_mode!")

        if axe is None:
            fig, axe = plt.subplots(figsize=figsize, sharex=True, sharey=True)
        else:
            try:
                fig = axe.get_figure()
            except Exception:
                fig = axe.flatten()[0].get_figure()

        cb = axe.imshow(im,
                        extent=[-spfx_extent, spfx_extent,
                                -spfx_extent, spfx_extent],
                        origin='lower', **kwargs)
        axe.set_xlim(-spfx_extent, spfx_extent)
        axe.set_xlabel("sp. freq. (cycles/rad)")

        if legend:
            txt = r"wl={:.4f}$\mu$m\n".format(wl*1e6)
            if 'color' not in kwargs_legend:
                kwargs_legend['color'] = "w"
            axe.text(0, 0.95*dim/2*pixSize, txt,
                     va="top", ha="center", **kwargs_legend)
        if colorbar:
            fig.colorbar(cb, ax=axe, label="Normalized Intensity")

        if savefig is not None:
            plt.savefig(savefig)
        return fig, axe, im
