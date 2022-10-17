"""Fourier

This script or rather the FFT class contained in it, takes a 2D-numpy arry and
applies the Fast Fourier Transform (FFT) to it for a certain wavelength and
returns the amplitude (either correlated flux, visibilities or squared
visibilities) and phase information.

This file can also be imported as a module and contains the following class:
    * FFT

Example of usage:
    Image in the form of a 2D-numpy array
    >>> image = np.array([[...], ...  [...]])
    >>> wavelength = ...
    >>> pixel_scale, zero_padding_order = ..., ...


    Initialise the FFT class
    >>> fft = FFT(image, wavelength, pixel_scale, zero_padding_order)

    Get the amplitude and phase of the FFT
    >>> amp, phase = fft.get_amp_phase(corr_flux=False, vis2=True)

    Plot the amplitude and phase information
    >>> fft.plot_amp_phase(matplot_axis=[], zoom=500,
                           corr_flux=False, uvcoords_lst=[], plt_save=True)

Credit
    FFT: https://numpy.org/doc/stable/reference/routines.fft.html

License
    Copyright (C) 2022 Marten Scheuck

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.units import Quantity
from numpy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.interpolate import interpn
from typing import List, Optional


class FastFourierTransform:
    """A collection and build up on the of the FFT-functionality provided by
    numpy

    ...
    """
    def __init__(self, image: Quantity, wavelength: Quantity,
                 pixel_scaling: float, zero_padding_order: Optional[int] = 1) -> None:
        self.model = image.value
        self.unpadded_model = image.copy()
        self.wl = wavelength
        self.pixel_scale = pixel_scaling
        self.zero_padding_order = zero_padding_order

        self.model_unpadded_dim = self.model.shape[0]
        self.model_unpadded_centre = self.model_unpadded_dim//2
        self.ft = self.do_fft2()

    @property
    def model_shape(self):
        """Fetches the model's x, and y shape"""
        return self.model.shape

    @property
    def dim(self):
        """Fetches the model's x-dimension.
        DISCLAIMER: Both dimensions are and need to be identical
        """
        return self.model_shape[0]

    @property
    def model_centre(self):
        """Fetches the model's centre"""
        return self.dim//2

    @property
    def freq_axis(self):
        """Fetches the FFT's frequency axis, scaled according to the
        pixel_scale (determined by the sampling and the FOV) as well as the
        zero padding factor"""
        return fftshift(fftfreq(self.zero_padding, self.pixel_scale))

    @property
    def fftscaling2m(self):
        """Fetches the FFT's scaling in meters"""
        cycles_per_rad = np.diff(self.freq_axis)[0].to(1/u.rad)
        return (cycles_per_rad*self.wl.to(u.m)).to(u.m, equivalencies=u.dimensionless_angles())

    @property
    def fftscaling2Mlambda(self):
        """Fetches the FFT's scaling in mega lambda"""
        return self.fftscaling2m/self.wl.to(u.m)

    @property
    def fftaxis_m_end(self):
        """Fetches the endpoint of the FFT's axis in meters"""
        return self.fftscaling2m*self.model_centre

    @property
    def fftaxis_Mlambda_end(self):
        """Fetches the endpoint of the FFT's axis in mega lambdas"""
        return self.fftscaling2Mlambda*self.model_centre

    @property
    def fftaxis_m(self):
        """Gets the FFT's axis's in [m]"""
        return np.linspace(-self.fftaxis_m_end, self.fftaxis_m_end,
                           self.dim, endpoint=False)

    @property
    def fftaxis_Mlambda(self):
        """Gets the FFT's axis's endpoints in [Mlambdas]"""
        return np.linspace(-self.fftaxis_Mlambda_end, self.fftaxis_Mlambda_end,
                           self.dim, endpoint=False)

    @property
    def grid_m(self):
        return self.fftaxis_m, self.fftaxis_m

    @property
    def zero_padding(self):
        """The new size of the image in [px] to be applied with zero padding"""
        return 2**int(np.log2(self.model_unpadded_dim)+self.zero_padding_order)

    def zero_pad_model(self):
        """This adds zero padding to the model image before it is transformed
        to increase the sampling in the FFT image
        """
        # TODO: Check if the zero padding moves the centre of the image -> Should be ok?
        padded_image = np.zeros((self.zero_padding, self.zero_padding))
        self.pad_centre = padded_image.shape[0]//2
        self.mod_min, self.mod_max = self.pad_centre-self.model_centre,\
                self.pad_centre+self.model_centre

        padded_image[self.mod_min:self.mod_max,
                     self.mod_min:self.mod_max] = self.model
        return padded_image

    def get_uv2fft2(self, uvcoords: Quantity, uvcoords_cphase: Quantity) -> Quantity:
        """Interpolates the input (u, v)-coordinates to the grid of the FFT

        Parameters
        ----------
        uvcoords: astropy.units.Quantity
            The (u, v)-coordinates of the instrument in [m] for the correlated
            fluxes/visibilities
        uvcoords_cphase: astropy.units.Quantity
            The (u, v)-coordinates of the instrument in [m] for the closure
            phases
        intp: bool, optional
            Decides if the coordinates are to be interpolated on the grid or
            rounded (interpolation is the favoured/better method and rounding
            was implemented as a debug tool)

        Returns
        -------
        amp: np.ndarray
            The interpolated amplitudes
        cphases: np.ndarray
            The interpolated closure phases
        xy_coords: List
            This list contains the indices for overplotting the uv-coordinates
            of the visibilities/correlated fluxes as well as the
            uv-coordinates for the closure phases to overplot on the phase plot
        """
        grid = tuple(axis.value for axis in self.grid_m)
        real_corr = interpn(grid, np.real(self.ft), uvcoords.value,
                            method='linear', bounds_error=False,
                            fill_value=None)
        imag_corr = interpn(grid, np.imag(self.ft), uvcoords.value,
                            method='linear', bounds_error=False,
                            fill_value=None)

        amp = np.abs(real_corr + 1j*imag_corr)
        real_phase = interpn(grid, np.real(self.ft), uvcoords_cphase.value,
                             method='linear', bounds_error=False,
                             fill_value=None)
        imag_phase = interpn(grid, np.imag(self.ft), uvcoords_cphase.value,
                             method='linear', bounds_error=False,
                             fill_value=None)
        cphases = np.angle(real_phase + 1j*imag_phase, deg=True)
        cphases = np.array(cphases).reshape(3, uvcoords_cphase.shape[1])

        cphases = np.sum(cphases, axis=0)
        # TODO: Check what impact this here has?
        cphases = np.degrees((np.radians(cphases) + np.pi) % (2*np.pi) - np.pi)
        return amp, cphases

    def get_amp_phase(self) -> List[Quantity]:
        """Gets the amplitude and the phase of the FFT

        Returns
        --------
        amp: astropy.units.Quantity
            The correlated fluxes or normed visibilities or visibilities
            squared pertaining to the function's settings
        phase: astropy.units.Quantity
            The phase information of the image after FFT
        """
        return np.abs(self.ft), np.angle(self.ft, deg=True)

    def do_fft2(self) -> Quantity:
        """Does the 2D-FFT and returns the 2D-FFT and shifts the centre to the
        middle

        Returns
        --------
        ft: np.ndarray
        """
        self.model = self.zero_pad_model()
        self.raw_fft = fft2(ifftshift(self.model))
        self.ft_centre = self.raw_fft[0][0]
        return fftshift(self.raw_fft)

    def plot_amp_phase(self, matplot_axis: Optional[List] = [],
                       zoom: Optional[int] = 500,
                       uv_coords: Optional[Quantity] = None,
                       uv_coords_cphase: Optional[Quantity] = None,
                       plt_save: Optional[bool] = False) -> None:
        """This plots the input model for the FFT as well as the resulting
        amplitudes and phases for units of both [m] and [Mlambda]

        Parameters
        ----------
        matplot_axis: List, optional
            The axis of matplotlib if the plot is to be embedded in a bigger
            plot
        zoom: bool, optional
            The zoom for the (u, v)-coordinates in [m], the [Mlambda] component
            will be automatically calculated to fit the axis
        uvcoords_lst: List, optional
            If not empty then the plots will be overplotted with the
            given (u, v)-coordinates
        plt_save: bool, optional
            Saves the plot if toggled on
        """
        if matplot_axis:
            fig, ax, bx, cx = matplot_axis
        else:
            fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
            ax, bx, cx = axarr.flatten()

        fov_scale = (self.pixel_scale*self.model_unpadded_dim).value
        fftaxis_m_end = self.fftaxis_m_end.value
        fftaxis_Mlambda_end = self.fftaxis_Mlambda_end.value
        zoom_Mlambda = ((zoom*u.dimensionless_unscaled)/(self.wl.to(u.m))).value

        vmax = (np.sort(self.unpadded_model.flatten())[::-1][1]).value

        amp, phase = self.get_amp_phase()
        ax.imshow(self.unpadded_model.value, vmax=vmax, interpolation="None",
                  extent=[-fov_scale, fov_scale, -fov_scale, fov_scale])
        cbx = bx.imshow(amp, extent=[-fftaxis_m_end, fftaxis_m_end-1,
                                     -fftaxis_Mlambda_end, fftaxis_Mlambda_end-1],
                        interpolation="None", aspect=(self.wl.to(u.m)).value)
        ccx = cx.imshow(phase, extent=[-fftaxis_m_end, fftaxis_m_end-1,
                                       -fftaxis_Mlambda_end, fftaxis_Mlambda_end-1],
                        interpolation="None", aspect=(self.wl.to(u.m)).value)

        fig.colorbar(cbx, fraction=0.046, pad=0.04, ax=bx, label="Flux [Jy]")
        fig.colorbar(ccx, fraction=0.046, pad=0.04, ax=cx, label="Phase [°]")

        ax.set_title(f"Model image at {self.wl}, Object plane")
        bx.set_title("Amplitude of FFT")
        cx.set_title("Phase of FFT")

        ax.set_xlabel("RA [mas]")
        ax.set_ylabel("DEC [mas]")

        bx.set_xlabel("u [m]")
        bx.set_ylabel(r"v [M$\lambda$]")

        cx.set_xlabel("u [m]")
        cx.set_ylabel(r"v [M$\lambda$]")

        bx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])
        cx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])

        fig.tight_layout()

        if plt_save:
            plt.savefig(f"{self.wl}_FFT_plot.png")
        else:
            if not matplot_axis:
                plt.show()


if __name__ == "__main__":
    ...
