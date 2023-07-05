import os
import neat
import datetime
import random
import visualize
import environment
import wblog



def run(config_file, run_config):
    random.seed(run_config['seed'])
    env = environment.Environment(run_config)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config.pop_size = env.pop_size
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Run for up to n generations.
    winner = p.run(env.evaluate_genomes, run_config['generations'])

    wblog.upload(stats, run_config)

    """
    node_names = {
        -1: "constant",
        -2: "vision 0",
        -3: "vision 1",
        -4: "vision 2",
        -5: "vision 3",
        -6: "vision 4",
        0: "out speed",
        1: "turn",
    }

    now = datetime.datetime.now()
    datestr = "%s%s%s_%s%s" % (now.year, now.month, now.day, now.hour, now.minute)

    os.mkdir("results/" + datestr)

    visualize.plot_stats(stats, ylog=False, view=True, filename=("results/" + datestr + "/avg_fitness_" + datestr + ".svg"))
    visualize.plot_species(stats, view=True, filename=("results/" + datestr + "/speciation_" + datestr + ".svg"))
    visualize.draw_net(config, winner, view=True, node_names=node_names, filename=("results/" + datestr +"/neural_net_" + datestr), fmt="png")

    # Find winner of last 10% of generations
    last_winners = stats.most_fit_genomes[-int((generations/10)):]
    last_winner = None

    max_fitness = 0
    for g in last_winners:
        if g.fitness > max_fitness:
            max_fitness = g.fitness
            last_winner = g

    visualize.draw_net(config, last_winner, view=True, node_names=node_names, filename=("results/" + datestr +"/last_neural_net_" + datestr), fmt="png")

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    """

if __name__ == '__main__':
    # visualize.draw_from_file('results/20221125_1133/avg_fitness_20221125_1133.svg.data')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    for i in range(10, 200, 10):
        run_config = {
            'generations' : 100,
            'grid_size' : 64,    # Size of the world
            'pop_size' : i,    # Initial population size
            'num_food' : 120,   # Initial amount of food
            'nutrition' : 200,   # Food nutrition
            'steps' : 300,   # Number of time steps per generation
            'seed' : 183
        }
        
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        run(config_path, run_config)
