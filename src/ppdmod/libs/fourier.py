import cv2
import scipy
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.units import Quantity
from typing import List, Optional

from .utils import _make_axis, make_fixed_params, make_delta_component,\
    make_ring_component, _make_params
from .combined_model import CombinedModel


class FastFourierTransform:
    """A collection and build up on the of the FFT-functionality provided by
    numpy

    ...
    """
    def __init__(self, image: Quantity, wavelength: Quantity,
                 pixel_scaling: Quantity, zero_padding_order: Optional[int] = 0) -> None:
        """"""
        self.wl = wavelength
        self.pixel_scaling = pixel_scaling

        self.unpadded_dim = image.shape[0]
        self.unpadded_centre = self.unpadded_dim//2

        self.fov_scale = (self.pixel_scaling*self.unpadded_centre).value
        self.model, self.unpadded_model, self.dim = self.zero_pad(image, zero_padding_order)
        self.model_centre = self.dim//2

        self.ft = self.get_fft()
        self.fftfreq_axis = self._get_fftfreq_axis(self.dim, self.pixel_scaling)
        self.meter_scaling = self.calculate_meter_scaling(self.fftfreq_axis)
        self.axis_meter_endpoint = self.meter_scaling*self.model_centre
        self.axis_Mlambda_endpoint = (self.meter_scaling/self.wl)*self.model_centre
        self.axis_m = _make_axis(self.axis_meter_endpoint, self.dim)
        self.axis_Mlambda = _make_axis(self.axis_Mlambda_endpoint, self.dim)
        self.grid_m = (self.axis_m, self.axis_m)

    # TODO: Finish docstring
    def _get_fftfreq_axis(self, dimension: int, sample_spacing: Quantity) -> Quantity:
        """Gets the winding frequency's axis

        Parameters
        ----------
        dimension: int
            The length of the input axis
        sample_spacing: astropy.units.Quantity
            The number of cycles that are passed with the winding frequency. The sampling
            spacing

        Returns
        -------
        freq_axis: Quantity
            The winding-frequency axis of the FFT [cycles/scaling.unit]
        """
        return np.fft.ifftshift(np.fft.fftfreq(dimension, d=sample_spacing))

    # TODO: Finish docstring
    def calculate_meter_scaling(self, fft_frequency_axis: Quantity) -> Quantity:
        """Calculates the meter scaling from the frequency axis given

        Parameters
        ----------
        fft_frequency_axis: Quantity
            The FFT frequency axis [cycles/astropy.units.m]

        Returns
        -------
        meter_scaling: astropy.units.Quantity
        """
        return (np.diff(fft_frequency_axis)[0].to(1/u.rad)).value*self.wl.to(u.m)

    def zero_pad(self, image: Quantity, zero_padding_order: int):
        """This adds zero padding to the model image before it is transformed
        to increase the sampling in the FFT image
        """
        dim = 2**int(np.log2(self.unpadded_dim)+zero_padding_order)
        new_dims = [dim//2-self.unpadded_centre]*4
        if zero_padding_order == 0:
            return image, image, dim

        padded_image = cv2.copyMakeBorder(image.value, *new_dims, cv2.BORDER_CONSTANT)
        return padded_image*image.unit, image, dim

    # TODO: Implement this
    def window(self):
        """Window function for the 2D-FFT"""
        ...

    def get_fft(self) -> np.ndarray:
        """Shifts the middle of the image to the top left and then vvaluates the two
        dimensional FFT before shifting it back

        Returns
        --------
        fourier_transform: np.ndarray
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.model)))

    def get_amp_phase(self, phase_wrap: Optional[bool] = False) -> List[Quantity]:
        """Gets the amplitude and the phase of the FFT

        Returns
        --------
        amp: astropy.units.Quantity
            The correlated fluxes or normed visibilities or visibilities
            squared pertaining to the function's settings
        phase: astropy.units.Quantity
            The phase information of the image after FFT
        """
        amp, phase = np.abs(self.ft), np.angle(self.ft, deg=True)
        if phase_wrap:
            phase = ((phase + np.pi) % (2 * np.pi) - np.pi)*-1
        return amp*self.model.unit, phase*u.deg

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
        real_corr = scipy.interpolate.interpn(grid, np.real(self.ft),
                                              uvcoords.value,
                                              method='linear', bounds_error=False,
                                              fill_value=None)
        imag_corr = scipy.interpolate.interpn(grid, np.imag(self.ft),
                                              uvcoords.value,
                                              method='linear', bounds_error=False,
                                              fill_value=None)

        amp = np.abs(real_corr + 1j*imag_corr)
        real_phase = scipy.interpolate.interpn(grid, np.real(self.ft),
                                               uvcoords_cphase.value,
                                               method='linear', bounds_error=False,
                                               fill_value=None)
        imag_phase = scipy.interpolate.interpn(grid, np.imag(self.ft),
                                               uvcoords_cphase.value,
                                               method='linear', bounds_error=False,
                                               fill_value=None)
        cphases = np.angle(real_phase + 1j*imag_phase, deg=True)*u.deg
        cphases = np.array(cphases).reshape(3, uvcoords_cphase.shape[1])
        cphases = np.sum(cphases, axis=0)
        # TODO: Check what impact this here has?
        cphases = np.degrees((np.radians(cphases) + np.pi) % (2*np.pi) - np.pi)
        return amp, cphases

    def plot_amp_phase(self, matplot_axes: Optional[List] = [],
                       zoom: Optional[int] = 500,
                       uv_coords: Optional[Quantity] = None,
                       uv_coords_cphase: Optional[Quantity] = None,
                       phase_wrap: Optional[bool] = False,
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
            Saves the plot if toggled on, else if not part of another plot, will show it
        """
        if matplot_axes:
            fig, ax, bx, cx = matplot_axes
        else:
            fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
            ax, bx, cx = axarr.flatten()

        axis_meter_endpoint = self.axis_meter_endpoint.value
        axis_Mlambda_endpoint = self.axis_Mlambda_endpoint.value
        zoom_Mlambda = zoom/self.wl.value

        vmax = (np.sort(self.unpadded_model.flatten())[::-1][1]).value

        amp, phase = self.get_amp_phase(phase_wrap=phase_wrap)
        ax.imshow(self.unpadded_model.value, vmax=vmax, interpolation="None",
                  extent=[-self.fov_scale, self.fov_scale, -self.fov_scale, self.fov_scale])
        cbx = bx.imshow(amp.value, extent=[-axis_meter_endpoint, axis_meter_endpoint,
                                           -axis_Mlambda_endpoint, axis_Mlambda_endpoint],
                        interpolation="None", aspect=self.wl.value)
        ccx = cx.imshow(phase.value, extent=[-axis_meter_endpoint, axis_meter_endpoint,
                                             -axis_Mlambda_endpoint, axis_Mlambda_endpoint],
                        interpolation="None", aspect=self.wl.value)

        fig.colorbar(cbx, fraction=0.046, pad=0.04, ax=bx, label="Flux [Jy]")
        fig.colorbar(ccx, fraction=0.046, pad=0.04, ax=cx, label="Phase [°]")

        ax.set_title(f"Model image at {self.wl}, Object plane")
        bx.set_title("Amplitude of FFT")
        cx.set_title("Phase of FFT")

        ax.set_xlabel(r"$\alpha$ [mas]")
        ax.set_ylabel(r"$\delta$ [mas]")

        bx.set_xlabel("u [m]")
        bx.set_ylabel(r"v [M$\lambda$]")

        cx.set_xlabel("u [m]")
        cx.set_ylabel(r"v [M$\lambda$]")

        bx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])
        cx.axis([-zoom, zoom, -zoom_Mlambda, zoom_Mlambda])

        fig.tight_layout()

        if uv_coords is not None:
            ucoord, vcoord = uv_coords[:, ::2].squeeze(), uv_coords[:, 1::2].squeeze()
            ucoord_cphase = [ucoords[:, ::2].squeeze() for ucoords in uv_coords_cphase]
            vcoord_cphase = [vcoords[:, 1::2].squeeze() for vcoords in uv_coords_cphase]
            vcoord, vcoord_cphase = map(lambda x: x/self.wl.value, [vcoord, vcoord_cphase])

            colors = np.array(["r", "g", "y"])
            bx.scatter(ucoord, vcoord, color="r")
            for i, ucoord in enumerate(ucoord_cphase):
                cx.scatter(ucoord, vcoord_cphase[i], color=colors[i])

        if plt_save:
            plt.savefig(f"{self.wl.value}-{self.wl.unit}_FFT_plot.png")
        else:
            if not matplot_axes:
                plt.show()


if __name__ == "__main__":
    fixed_params = make_fixed_params(30, 128, 1500, 7900, 140, 19, 1)
    disc_params = _make_params([1., 1.],
                               [u.dimensionless_unscaled, u.dimensionless_unscaled],
                               ["q", "p"])
    geometric_params = _make_params([0.5, 140], [u.dimensionless_unscaled, u.deg],
                                    ["axis_ratio", "pa"])
    modulation_params = _make_params([0.5, 140], [u.dimensionless_unscaled, u.deg],
                                     ["mod_amp", "mod_angle"])
    wavelengths = [8]*u.um
    wavelength = 8*u.um
    complete_ring = make_ring_component("inner_ring",
                                        params=[0., 0., 5., 0.])
    delta_component = make_delta_component("star")

    model = CombinedModel(fixed_params, disc_params, wavelengths,
                          geometric_params, modulation_params)
    model.add_component(complete_ring)
    model.add_component(delta_component)
    image = model.eval_flux(wavelength)
    np.save("model.npy", image.value)
    fourier = FastFourierTransform(image, wavelength,
                                   model.pixel_scaling, 3)
    print(model.eval_total_flux(wavelength))
    fourier.plot_amp_phase(zoom=500)

