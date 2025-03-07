"""Model diffusion through random motion of particles."""

import agentpy as ap


class Particle(ap.Agent):
    pass


class DiffusionModel(ap.Model):

    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, Particle)
        self.grid = ap.Grid(self, (self.p.n_rows, self.p.n_cols))

        positions = (self.p.initial_location,) * self.p.agents
        self.grid.add_agents(self.agents, positions=positions)
