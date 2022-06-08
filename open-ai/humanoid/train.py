import gym
import neat
import numpy as np
import pickle
import time
import multiprocessing
import numpy as np
import utils

NUM_GENERATIONS = 10000

global NUM_ENVS_VALIDATION
NUM_ENVS_VALIDATION = 3

SAVE_STEP = 10


def eval_genome(genome, config):
    global BEST_FITNESS

    envs = gym.vector.SyncVectorEnv([
    lambda: gym.make("Humanoid-v4") for i in range(NUM_ENVS_VALIDATION)
    ])

    observation = envs.reset(seed=np.random.randint(0,500))
    manager = utils.reward_manager(1.005)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    current_fitness = np.zeros(NUM_ENVS_VALIDATION)

    done = False
    while not done:
        action = [net.activate(ob) for ob in observation]
        action = np.array(action)/2.5 # range -0.4 to 0.4
        observation, reward, done, info = envs.step(action)
        current_fitness = manager.step(current_fitness, reward)
        done = np.any(done)

    genome.fitness = np.average(current_fitness)
    envs.close()
    return genome.fitness


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat.cfg')
p = neat.Population(config)
stats = neat.StatisticsReporter()
# p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(stats)
p.add_reporter(utils.StdOutReporter_mode(True))

pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

for i in range(NUM_GENERATIONS):
    genome = p.run(pe.evaluate, SAVE_STEP)
    best = stats.best_genome()
    with open('result/best_%d.pkl'%(i*SAVE_STEP), 'wb') as output:
        pickle.dump(best, output, 1)