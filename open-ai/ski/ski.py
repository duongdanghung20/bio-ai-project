import gym
import neat
import numpy as np
import pickle
import time
import cv2
import multiprocessing

"""
UNDER DEVELOPEMENT, DOES NOT RUN CORRECTLY
"""

NUM_GENERATIONS = 50

env = gym.make("ALE/Skiing-v5", full_action_space=False, obs_type='grayscale')
env.fps = 60
print(env.action_space, len(env.observation_space.high))


def eval_genome(genome, config):

    observation = env.reset()

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    current_fitness = 0 

    done = False
    counter = 0
    while not done:

        inx, iny = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        observation = cv2.resize(observation, (inx, iny))
        observation = np.ndarray.flatten(observation)
        output = net.activate(observation)
        action = np.argmax(output)

        observation, reward, done, info = env.step(action=action)

        current_fitness += reward
        counter += 1

        
        #time.sleep(0.001)
    genome.fitness = current_fitness
    return genome.fitness
    #env.close()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedfw.txt')
p = neat.Population(config)

#p.add_reporter(neat.Checkpointer(10))

stats = neat.StatisticsReporter()
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(stats)

pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
winner = p.run(pe.evaluate, NUM_GENERATIONS)
print(winner)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)  


