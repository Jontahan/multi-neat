import os

import neat
import datetime
import params
import visualize

from environment import Environment

env = Environment()


# 2-input XOR inputs and expected outputs.
def eval_genomes(genomes, config):
    env.reset()
    env.render()

    for genome_id, genome in genomes:
        genome.fitness = 0

    for i in range(params.EPISODE_LENGTH):
        for (genome_id, genome), indiv in zip(genomes, env.individuals):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            output = net.activate(indiv.inputs(env))
            indiv.execute_action(output)
            for food in env.foods:
                if abs(indiv.pos[0] - food.position[0]) < 0.5 and abs(indiv.pos[1] - food.position[1]) < 0.5:
                    indiv.energy += food.nutrition
                    env.foods.remove(food)
            genome.fitness = indiv.energy

        if (env.state + 1) % 100 == 0:
            env.render()
    env.state += 1

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.pop_size = params.NUM_IND
    config.fitness_threshold = params.FITNESS_THRESHOLD
    print(f"config: {config.pop_size}")

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to n generations.
    winner = p.run(eval_genomes, 400)

    node_names = {
        -1: "constant",
        -2: "in speed",
        -3: "angle",
        -4: "angle to food",
        -5: "age",
        0: "out speed",
        1: "turn",
    }

    now = datetime.datetime.now()
    datestr = "%s%s%s_%s%s" % (now.year, now.month, now.day, now.hour, now.minute)

    visualize.plot_stats(stats, ylog=False, view=True, filename=("results/avg_fitness_" + datestr + ".svg"))
    visualize.plot_species(stats, view=True, filename=("results/speciation_" + datestr + ".svg"))
    visualize.draw_net(config, winner, view=True, node_names=node_names, filename=("results/neural_net_" + datestr), fmt="png")

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
