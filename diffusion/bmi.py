# -*- coding: utf-8 -*-
import pathlib
from collections import namedtuple

import numpy
from bmipy import Bmi

from .diffusion_model import DiffusionModel
from .io import load_config

HERE = pathlib.Path(__file__)
MODULE_PATH = HERE.parent

BmiVar = namedtuple(
    "BmiVar", ["dtype", "itemsize", "nbytes", "units", "location", "grid"]
)
BmiGridUniformRectilinear = namedtuple(
    "BmiGridUniformRectilinear", ["shape", "yx_spacing", "yx_of_lower_left"]
)


class BmiDiffusionModel(Bmi):
    """Model diffusion stochastically on a 2D domain."""

    _name = "AgentPy Diffusion Model"
    _input_var_names = ()
    _output_var_names = ("model_grid__histogram_of_agents",)

    def __init__(self):
        self._config = {}
        self._model = None
        self._var = None
        self._grid = {}
        self._time = {
            "current": 0.0,
            "start": 0.0,
            "end": numpy.finfo("d").max,
            "units": "1",
            "step": 1.0,
        }

    def finalize(self) -> None:
        self._model.end()
        self._model = None

    def get_component_name(self) -> str:
        return self._name

    def get_current_time(self) -> float:
        return self._time["current"]

    def get_end_time(self) -> float:
        return self._time["end"]

    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(
        self, grid: int, edge_nodes: numpy.ndarray
    ) -> numpy.ndarray:
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_edges(
        self, grid: int, face_edges: numpy.ndarray
    ) -> numpy.ndarray:
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_face_nodes(
        self, grid: int, face_nodes: numpy.ndarray
    ) -> numpy.ndarray:
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid: int) -> int:
        return self.get_grid_size(grid)

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: numpy.ndarray
    ) -> numpy.ndarray:
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_origin(self, grid: int, origin: numpy.ndarray) -> numpy.ndarray:
        origin[:] = self._grid[grid].yx_of_lower_left
        return origin

    def get_grid_rank(self, grid: int) -> int:
        return len(self._grid[grid].shape)

    def get_grid_shape(self, grid: int, shape: numpy.ndarray) -> numpy.ndarray:
        shape[:] = self._grid[grid].shape
        return shape

    def get_grid_size(self, grid: int) -> int:
        return int(numpy.prod(self._grid[grid].shape))

    def get_grid_spacing(self, grid: int, spacing: numpy.ndarray) -> numpy.ndarray:
        spacing[:] = self._grid[grid].yx_spacing
        return spacing

    def get_grid_type(self, grid: int) -> str:
        return "uniform_rectilinear"

    def get_grid_x(self, grid: int, x: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid: int, y: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid: int, z: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("get_grid_z")

    def get_input_item_count(self) -> int:
        return len(self._input_var_names)

    def get_input_var_names(self) -> tuple[str]:
        return self._input_var_names

    def get_output_item_count(self) -> int:
        return len(self._output_var_names)

    def get_output_var_names(self) -> tuple[str]:
        return self._output_var_names

    def get_start_time(self) -> float:
        return self._time["start"]

    def get_time_step(self) -> float:
        return self._time["step"]

    def get_time_units(self) -> str:
        return self._time["units"]

    def get_value(self, name: str, dest: numpy.ndarray) -> numpy.ndarray:
        dest[:] = self._model.histogram.flatten()
        return dest

    def get_value_at_indices(
        self, name: str, dest: numpy.ndarray, inds: numpy.ndarray
    ) -> numpy.ndarray:
        raise NotImplementedError("get_value_at_indices")

    def get_value_ptr(self, name: str) -> numpy.ndarray:
        raise NotImplementedError("get_value_ptr")

    def get_var_grid(self, name: str) -> int:
        return self._var.grid

    def get_var_itemsize(self, name: str) -> int:
        return self._var.itemsize

    def get_var_location(self, name: str) -> str:
        return self._var.location

    def get_var_nbytes(self, name: str) -> int:
        return self._var.nbytes

    def get_var_type(self, name: str) -> str:
        return self._var.dtype

    def get_var_units(self, name: str) -> str:
        return self._var.units

    def initialize(self, config_file: str) -> None:
        self._config = load_config(config_file)

        self._model = DiffusionModel(self._config)
        self._model.sim_setup()

        self._time["current"] = float(self._model.t)
        self._time["end"] = float(self._model.p.steps)

        self._var = BmiVar(
            dtype=str(self._model.histogram.dtype),
            itemsize=self._model.histogram.itemsize,
            nbytes=self._model.histogram.nbytes,
            location="face",
            units="1",
            grid=0,
        )

        self._grid = {
            0: BmiGridUniformRectilinear(
                shape=self._model.histogram.shape,
                yx_spacing=(
                    1.0,
                    1.0,
                ),
                yx_of_lower_left=(
                    0.0,
                    0.0,
                ),
            )
        }

    def set_value(self, name: str, src: numpy.ndarray) -> None:
        raise NotImplementedError("set_value")

    def set_value_at_indices(
        self, name: str, inds: numpy.ndarray, src: numpy.ndarray
    ) -> None:
        raise NotImplementedError("set_value_at_indices")

    def update(self) -> None:
        self._model.sim_step()
        self._time["current"] = float(self._model.t)

    def update_until(self, time: float) -> None:
        raise NotImplementedError("update_until")
