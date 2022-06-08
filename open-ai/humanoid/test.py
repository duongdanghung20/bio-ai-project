import neat
import gym
import pickle
import os
import time
import numpy as np
import argparse
import utils
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, default="best.pkl")
args = parser.parse_args()

env = gym.make("Humanoid-v4")

with open(args.i, 'rb') as file:
    genome = pickle.load(file)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'neat.cfg')

net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

print("================== start ================== ")
print("\ngenome: ", genome.fitness, "\n")
result = []
for _ in tqdm(range(100)):
    manager = utils.reward_manager(1)
    # if _ < 8: continue
    current_fitness = 0 
    done = False
    observation = env.reset(seed=_)
    # observation = env.reset(seed=np.random.randint(0,500))

    while not done:
        action = net.activate(observation)
        action = np.array(action)/2.5 # range -0.4 to 0.4

        observation, reward, done, info = env.step(action)

        current_fitness = manager.step(current_fitness, reward)
        
        # time.sleep(0.01)
        # env.render()
    result.append(current_fitness)
print(np.average(result))