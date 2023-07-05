import numpy as np
import neat

from indiv import Indiv
from renderer import Renderer


class Environment:
    agents = []
    foods = []

    def __init__(self, run_config):
        self.rand = np.random.default_rng(run_config['seed'])
        self.grid_size = run_config['grid_size']
        self.pop_size = run_config['pop_size']
        self.num_food = run_config['num_food']
        self.nutrition = run_config['nutrition']
        self.steps = run_config['steps']
        
        self.state = 0
        self.renderer = Renderer(self.grid_size)



    def evaluate_genomes(self, genomes, config):
        self.state += 1
        self.agents = []
        self.foods = []
        # Initialize
        for genome_id, genome in genomes:
            # Create a brain
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Create an agent and connect the brain
            x = self.rand.random() * (self.grid_size - 1) + 0.5
            y = self.rand.random() * (self.grid_size - 1) + 0.5
            a = self.rand.random() * 2 * np.pi - np.pi
            agent = Indiv(x, y, a, net)
            self.agents.append(agent)

        # Simulate
        self.foods = self.rand.random((self.num_food, 2)) * (self.grid_size - 4) + 2  # Spawn food
        for i in range(self.steps):
            for agent in self.agents:
                # Agent acts
                agent.step(self)

                # Check food
                new_foods = np.delete(self.foods, np.where(
                    (abs(self.foods[:, 0] - agent.x) < 0.5) &
                    (abs(self.foods[:, 1] - agent.y) < 0.5))[0], axis=0)
                agent.energy += self.nutrition * (len(self.foods) - len(new_foods))
                self.foods = new_foods

            # Render world to screen
            if self.state % 100 == 0:
                self.renderer.render(self)

        # Evaluate
        for (genome_id, genome), agent in zip(genomes, self.agents):
            genome.fitness = agent.energy


