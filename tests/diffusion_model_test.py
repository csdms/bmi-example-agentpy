"""Test the diffusion_model module."""

import agentpy as ap
from diffusion import DiffusionModel


def test_model_is_a_diffusionmodel():
    m = DiffusionModel()
    assert isinstance(m, DiffusionModel)


def test_model_is_an_agentpy_model():
    m = DiffusionModel()
    assert isinstance(m, ap.Model)
