import neat
import gym
import pickle
import os
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, default="best.pkl")
args = parser.parse_args()

class reward_manager():
    def __init__(self):
        self.num_step = 1
        self.factor = 0.995

    def reset(self ):
        self.__init__()

    def step(self, value, reward):
        value = value + (self.factor**self.num_step)*reward
        self.num_step+=1
        return value


with open(args.i, 'rb') as file:
    genome = pickle.load(file)

# Load the config file
local_dir = os.path.dirname(__file__)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'neat.cfg')

# Simulate the genome genome
net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

print("================== start ================== ")
env = gym.make("Humanoid-v4")
print("\ngenome: ", genome.fitness, "\n")

for _ in range(100):

    manager = reward_manager()
    # if _ < 8: continue
    current_fitness = 0 
    done = False
    observation = env.reset(seed=np.random.randint(0,500))
    while not done:
        action = net.activate(observation)
        action = np.array(action)/2.5 # range -0.4 to 0.4

        observation, reward, done, info = env.step(action)

        current_fitness = manager.step(current_fitness, reward)
        # current_fitness += reward
        
        time.sleep(0.01)
        genome.fitness = current_fitness
        env.render()
    print(genome.fitness)
