import gym
import neat
import numpy as np
import pickle
import time
import os
import multiprocessing
import datetime
import visualize

"""
code for running the open-ai gym biperdalwalker environment and training
an agent using NEAT-algorithm. Specify the paths and run this file
To change parameters, use the 'config-bipedal' text file. 

"""

CWD = os.getcwd()
CHECKPOINTS_PATH = os.path.join(CWD, 'checkpoints/08_06_recurrent/')
GENOMES_PATH = os.path.join(CWD, 'best_genomes/08_06_recurrent/')
FIGURES_PATH = os.path.join(CWD, 'figures/08_06_recurrent/')

os.makedirs(GENOMES_PATH)
os.makedirs(CHECKPOINTS_PATH)
os.makedirs(FIGURES_PATH)

NUM_GENERATIONS = 200


global BEST_FITNESS
BEST_FITNESS=-100

def eval_genome(genome, config):
    global BEST_FITNESS

    env = gym.make("BipedalWalker-v3")
    observation = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    current_fitness = 0 

    done = False
    while not done:
        output = net.activate(observation)
        action = output

        observation, reward, done, info = env.step(action=action)

        current_fitness += reward

        if current_fitness >= BEST_FITNESS:
            BEST_FITNESS = current_fitness
            #env.render()
        
    genome.fitness = current_fitness
    return genome.fitness


def save_genomes(genomes, genome_names):
    current_time = datetime.datetime.now().strftime('%m_%d_%Y_%H:%M:%S')
    for i, genome in enumerate(genomes):
        with open(os.path.join(GENOMES_PATH, str(i)+'_genome'+current_time+'.pkl'), 'wb') as f:
            pickle.dump(genome, f, 1)  

def make_plots(winner, stats):
    raise NotImplementedError
    

def main():
    #Define if continue from a checkpoint or not
    load = False
    
    genomes_to_save = []

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-bipedal.txt')
    p = neat.Population(config)
    if load:
        p = neat.Checkpointer.restore_checkpoint('/home/ilmarilehtinen/code/git_repos/bio-ai-project/open-ai/bipedal/checkpoints/07_06_avg/189')
    p.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=CHECKPOINTS_PATH))

    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)


    parallel = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(parallel.evaluate, NUM_GENERATIONS)

    #save 5 best genomes from each run
    n_best = stats.best_genomes(5)
    names = [f'{i}_genome' for i in range(1,6)]
    genomes_to_save.extend(n_best)
    save_genomes(genomes_to_save, names)

    #plots
    #make_plots(winner, stats)
    visualize.draw_net(config, winner, view=False, node_names=None, filename=FIGURES_PATH + "net")
    visualize.plot_stats(stats, ylog=False, view=False, filename=FIGURES_PATH + "fitness.svg")
    visualize.plot_species(stats, view=False, filename=FIGURES_PATH + "species.svg")



if __name__ == '__main__':
    main()


