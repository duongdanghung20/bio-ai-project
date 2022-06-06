import gym
import neat
import numpy as np
import pickle
import time
import multiprocessing

NUM_GENERATIONS = 50

env = gym.make("BipedalWalker-v3")

env.fps = 60

global BEST_FITNESS
BEST_FITNESS=-100

def eval_genome(genome, config):
    global BEST_FITNESS

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

best = stats.best_genome()
print(best)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)  


