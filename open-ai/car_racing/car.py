import gym
import neat
import numpy as np
import pickle
import time
import multiprocessing
import visualize
import os
import cv2

NUM_GENERATIONS = 50

env = gym.make("CarRacing-v1")

env.fps = 60

CWD = os.getcwd()
CHECKPOINTS_PATH = os.path.join(CWD, 'checkpoints/')
GENOMES_PATH = os.path.join(CWD, 'best_genomes/')
FIGURES_PATH = os.path.join(CWD, 'figures/')

def eval_genome(genome, config):
    observation = env.reset()

    net = neat.nn.RecurrentNetwork.create(genome, config)
    current_fitness = 0 
    inx, iny, inc = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)

    done = False
    while not done:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.ndarray.flatten(observation)
        
        output = net.activate(observation)

        observation, reward, done, info = env.step(output)

        current_fitness += reward
        
        time.sleep(0.001)
        genome.fitness = current_fitness
    #env.close()

    #print(current_fitness)
    return genome.fitness

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-car.txt')
p = neat.Population(config)

#p.add_reporter(neat.Checkpointer(10))

stats = neat.StatisticsReporter()
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(stats)

pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
winner = p.run(pe.evaluate, NUM_GENERATIONS)
print(winner)

best = stats.best_genome()


with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)  

with open('best.pkl', 'wb') as output:
    pickle.dump(best, output, 1)  

visualize.draw_net(config, winner, view=False, node_names=None, filename=FIGURES_PATH + "net")
visualize.plot_stats(stats, ylog=False, view=False, filename=FIGURES_PATH + "fitness.svg")
visualize.plot_species(stats, view=False, filename=FIGURES_PATH + "species.svg")


