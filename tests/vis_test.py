"""Test the vis module."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.image import AxesImage

from diffusion import DiffusionModel, load_config
from diffusion.vis import (animation_colorbar_plot, has_colorbar,
                           histogram_colorbar_plot, histogram_plot)


@pytest.fixture
def model(shared_datadir):
    m = DiffusionModel(load_config(shared_datadir / "config.yaml"))
    m.run()
    return m


def test_histogram_plot(model):
    fig, ax = plt.subplots()
    hp = histogram_plot(model, ax)
    assert isinstance(hp, AxesImage)


def test_histogram_colorbar_plot(model):
    fig, ax = plt.subplots()
    histogram_colorbar_plot(model, ax, fig)
    assert has_colorbar(fig) is True


def test_animation_colorbar_plot(model):
    fig, ax = plt.subplots()
    animation_colorbar_plot(model, ax, fig)
    assert has_colorbar(fig) is True
