from typing import Optional, List
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from .options import OPTIONS


def plot_corner(sampler: np.ndarray,
                wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    corner.corner(samples, show_titles=True,
                  labels=OPTIONS["model.params"].keys(),
                  plot_datapoints=True,
                  quantiles=[0.16, 0.5, 0.84])

    if save_path:
        plt.savefig(
            Path(save_path) / f"Corner_plot_{(wavelength):.2f}.png")
    else:
        plt.show()


def plot_chains(sampler: np.ndarray,
                theta: List,
                wavelength: float,
                save_path: Optional[str] = None) -> None:
    """Plots the chains for debugging to see if and how they converge"""
    _, axes = plt.subplots(len(theta), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    ndim = len(theta)

    for index in range(ndim):
        axes[index].plot(samples[:, :, index], "k", alpha=0.3)
        axes[index].set_xlim(0, len(samples))
        axes[index].set_ylabel(OPTIONS["model.params"].keys()[index])
        axes[index].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    if save_path:
        plt.savefig(
            Path(save_path) / f"Chain_plot_{(wavelength):.2f}.png")
    else:
        plt.show()
