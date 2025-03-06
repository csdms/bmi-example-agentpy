"""Test the io module."""

from diffusion.io import load_config

CONFIG_FILE = "config.yaml"


def test_load_config(shared_datadir):
    params = load_config(shared_datadir / CONFIG_FILE)
    assert "agents" in params.keys()
    assert params["agents"] == 100
