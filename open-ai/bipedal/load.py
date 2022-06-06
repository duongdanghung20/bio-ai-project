import neat
import gym
import pickle
import os
import time
import numpy as np

winner_path = '/home/ilmarilehtinen/code/git_repos/bio-ai-project/open-ai/bipedal/'


with open(winner_path + 'winner.pkl', 'rb') as file:
    winner = pickle.load(file)
    print("\nWinner: ", winner, "\n")

# Load the config file
local_dir = os.path.dirname(__file__)
#config_file = os.path.join(local_dir, CONFIG_PATHS[ENV_ID])
config_file = '/home/ilmarilehtinen/code/git_repos/bio-ai-project/open-ai/bipedal/config-feedfw.txt'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_file)

# Simulate the winner genome
net = neat.nn.FeedForwardNetwork.create(genome=winner, config=config)

env = gym.make("BipedalWalker-v3")
for _ in range(20):
    current_fitness = 0 

    done = False
    observation = env.reset()
    while not done:
        output = net.activate(observation)
        action = output

        observation, reward, done, info = env.step(action)

        current_fitness += reward
        
        time.sleep(0.01)
        winner.fitness = current_fitness
        env.render()
    print(winner.fitness)
#env.close()

