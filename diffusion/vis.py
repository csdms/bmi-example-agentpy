"""Visualize DiffusionModel outputs."""

import agentpy as ap
import matplotlib.ticker as mticker


def histogram_plot(model, ax):
    """Display the histogram of particles."""
    im = ap.gridplot(model.histogram, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Diffusion model (t = {model.t})")

    return im


def histogram_colorbar_plot(model, ax, fig):
    """Display the histogram of particles with a colorbar."""
    im = histogram_plot(model, ax)
    fig.colorbar(im, ax=ax, label="Particle count", shrink=0.8)


def animation_colorbar_plot(model, ax, fig):
    """Display the histogram of particles with a colorbar for animation."""
    im = histogram_plot(model, ax)
    fig.colorbar(
        im,
        ax=ax,
        label="Particle count",
        shrink=0.8,
        ticks=[0, model.p.agents // 2, model.p.agents],
        format=mticker.FixedFormatter(["few", "some", "many"]),
    )
