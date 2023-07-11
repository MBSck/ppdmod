from typing import Optional, List

from astropy.units import Quantity


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
              extent=[-self.fov, self.fov, -self.fov, self.fov])
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


