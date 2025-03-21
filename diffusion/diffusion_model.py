"""Model diffusion through random motion of particles."""

import agentpy as ap
import numpy as np

BASE_WEIGHT = 10

directions = [
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
]
weights = np.ones(len(directions), dtype=int) * BASE_WEIGHT


class Particle(ap.Agent):

    def setup(self):
        self.displacement = [0, 0]

    def set_random_displacement(self):
        self.displacement = self.model.random.choices(directions, weights=weights).pop()


class DiffusionModel(ap.Model):

    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, Particle)
        self.grid = ap.Grid(self, (self.p.n_rows, self.p.n_cols))

        positions = (self.p.initial_location,) * self.p.agents
        self.grid.add_agents(self.agents, positions=positions)

        self.set_diffusivity()
        self.histogram = None

    def set_diffusivity(self):
        weights[1:] = self.p.diffusivity

    def update(self):
        self.histogram = self.grid.apply(len, field="agents")
        self.agents.set_random_displacement()

    def step(self):
        for agent in self.agents:
            self.grid.move_by(agent, agent.displacement)
