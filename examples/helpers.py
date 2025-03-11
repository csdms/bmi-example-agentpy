"""Helper functions for working with DiffusionModel."""

import agentpy as ap


def histogram_plot(model, ax):
    """Display the histogram of particles."""
    im = ap.gridplot(model.histogram, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Diffusion model (t = {model.t})")

    return im


def histogram_colorbar_plot(model, ax, fig):
    im = histogram_plot(model, ax)
    fig.colorbar(im, ax=ax, label="Particle count", shrink=0.8)
