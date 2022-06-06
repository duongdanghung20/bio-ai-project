import gym
import neat
import numpy as np
import pickle
import time
import multiprocessing

NUM_GENERATIONS = 100

env = gym.make("LunarLander-v2")

env.fps = 60

def eval_genome(genome, config):
    observation = env.reset()

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    current_fitness = 0 

    done = False
    while not done:
        output = net.activate(observation)
        action = np.argmax(output)

        observation, reward, done, info = env.step(action)

        current_fitness += reward
        
        time.sleep(0.001)
        genome.fitness = current_fitness
    #env.close()

    #print(current_fitness)
    return genome.fitness

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


