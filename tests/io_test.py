"""Test the io module."""

from diffusion.io import load_config

CONFIG_FILE = "config.yaml"


def test_load_config(shared_datadir):
    params = load_config(shared_datadir / CONFIG_FILE)
    assert "n_agents" in params.keys()
    assert params["n_agents"] == 100
