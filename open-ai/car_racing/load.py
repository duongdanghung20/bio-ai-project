import neat
import gym
import pickle
import os
import time
import numpy as np
import cv2

CWD = os.getcwd()
winner_path = CWD

with open(winner_path + '/best.pkl', 'rb') as file:
    winner = pickle.load(file)
    print("\nWinner: ", winner, "\n")

# Load the config file
local_dir = os.path.dirname(__file__)
#config_file = os.path.join(local_dir, CONFIG_PATHS[ENV_ID])
config_file = './config-car.txt'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_file)

# Simulate the winner genome
net = neat.nn.RecurrentNetwork.create(genome=winner, config=config)

env = gym.make('CarRacing-v1')
for _ in range(20):
    current_fitness = 0 
    inx, iny, inc = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)

    done = False
    observation = env.reset()
    while not done:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.ndarray.flatten(observation)
        output = net.activate(observation)

        observation, reward, done, info = env.step(observation)

        current_fitness += reward
        
        time.sleep(0.01)
        winner.fitness = current_fitness
        env.render()
    print(winner.fitness)
#env.close()

