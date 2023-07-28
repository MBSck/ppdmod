from typing import Optional, List

from astropy.units import Quantity


def plot_corner(sampler: np.ndarray,
                labels: List, wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                       plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    plot_name = f"Corner_plot_{(wavelength):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))


def plot_chains(sampler: np.ndarray, theta: List,
                labels: List, wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the chains for debugging to see if and how they converge"""
    fig, axes = plt.subplots(len(theta), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    ndim = len(theta)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plot_name = f"Chain_plot_{(wavelength):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))
