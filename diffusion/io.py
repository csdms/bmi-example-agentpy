"""Read/write to files."""

import yaml


def load_config(config_file):
    try:
        with open(config_file, "r") as fp:
            return yaml.safe_load(fp).get("DiffusionModel", {})
    except FileNotFoundError:
        raise
