"""Test the diffusion_model module."""

import agentpy as ap
from diffusion import DiffusionModel

PARAMETERS = {
    "agents": 100,
    "steps": 20,
    "n_cols": 5,
    "n_rows": 5,
    "bmi_version": "2.0",
}


def test_model_is_a_diffusionmodel():
    m = DiffusionModel()
    assert isinstance(m, DiffusionModel)


def test_model_is_an_agentpy_model():
    m = DiffusionModel()
    assert isinstance(m, ap.Model)


def test_model_has_parameters():
    m = DiffusionModel(PARAMETERS)
    assert m.p.agents == PARAMETERS["agents"]
    assert m.p.steps == PARAMETERS["steps"]
    assert m.p.n_cols == PARAMETERS["n_cols"]
    assert m.p.n_rows == PARAMETERS["n_rows"]
