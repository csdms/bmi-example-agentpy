"""Model diffusion through random motion of particles."""

import agentpy as ap


class Particle(ap.Agent):

    direction = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ]

    def setup(self):
        self.displacement = [0, 0]

    def set_random_displacement(self):
        self.displacement = self.model.random.choice(Particle.direction)


class DiffusionModel(ap.Model):

    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, Particle)
        self.grid = ap.Grid(self, (self.p.n_rows, self.p.n_cols))

        positions = (self.p.initial_location,) * self.p.agents
        self.grid.add_agents(self.agents, positions=positions)

        self.histogram = None

    def update(self):
        self.histogram = self.grid.apply(len, field="agents")
        self.agents.set_random_displacement()

    def step(self):
        for agent in self.agents:
            self.grid.move_by(agent, agent.displacement)
