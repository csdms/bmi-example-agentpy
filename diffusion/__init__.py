"""Model diffusion over a 2D domain."""

from .diffusion_model import DiffusionModel
from .io import load_config

__all__ = ["DiffusionModel", "load_config"]
