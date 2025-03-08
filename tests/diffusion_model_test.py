"""Test the diffusion_model module."""

import math
import agentpy as ap
from diffusion import DiffusionModel

PARAMETERS = {
    "agents": 40,
    "steps": 20,
    "n_cols": 6,
    "n_rows": 5,
    "initial_location": [2, 2],
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


def test_model_setup():
    m = DiffusionModel(PARAMETERS)
    m.setup()
    assert len(m.agents) == PARAMETERS["agents"]
    assert m.grid.shape == (PARAMETERS["n_rows"], PARAMETERS["n_cols"])
    assert len(m.grid.grid[tuple(PARAMETERS["initial_location"])][0]) == PARAMETERS["agents"]


def test_particle_displacement():
    m = DiffusionModel(PARAMETERS)
    m.setup()
    initial_positions = list(m.grid.positions.values())
    m.run(steps=1)
    final_positions = list(m.grid.positions.values())
    for i in range(len(m.agents)):
        assert math.dist(initial_positions[i], final_positions[i]) <= 1.0
