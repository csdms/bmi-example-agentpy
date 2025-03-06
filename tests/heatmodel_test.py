"""Test the heatmodel module."""

import agentpy as ap
from heat import HeatModel


def test_model_is_a_heatmodel():
    m = HeatModel()
    assert isinstance(m, HeatModel)


def test_model_is_an_agentpy_model():
    m = HeatModel()
    assert isinstance(m, ap.Model)
