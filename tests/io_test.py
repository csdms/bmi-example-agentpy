"""Test the io module."""

import pytest

from diffusion.io import load_config

CONFIG_FILE = "config.yaml"


def test_load_config(shared_datadir):
    params = load_config(shared_datadir / CONFIG_FILE)
    assert "agents" in params.keys()
    assert params["agents"] == 100


def test_load_config_fails_on_unknown_file():
    with pytest.raises(FileNotFoundError):
        _ = load_config("this-file-does-not-exist.yaml")
