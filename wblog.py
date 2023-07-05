import wandb 
import numpy as np

def upload(statistics, config):
    wandb.init(project='multi-neat', config=config)

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())

    for i in generation:
        wandb.log(step=i, data={
            'best_fitness': best_fitness[i], 
            'avg_fitness': avg_fitness[i]
        })
    
    wandb.finish()