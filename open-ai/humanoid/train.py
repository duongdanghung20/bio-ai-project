import gym
import neat
import numpy as np
import pickle
import time
import multiprocessing
import numpy as np
import json
from neat.math_util import mean, stdev

NUM_GENERATIONS = 10000

global BEST_FITNESS
BEST_FITNESS=-100

class StdOutReporter_mode(neat.StdOutReporter):
  def __init__(self, show_species_detail):
    super().__init__(show_species_detail)
    self.log = []

  def post_evaluate(self, config, population, species, best_genome):
    fitnesses = [c.fitness for c in population.values()]
    fit_mean = mean(fitnesses)
    fit_std = stdev(fitnesses)
    best_species_id = species.get_species_id(best_genome.key)
    
    tmp = {"Pop_avg_fitness":fit_mean,
           "stdev": fit_std,
        #    "adj_fitness":max_adjusted_fitness,
           "Best_fitness":best_genome.fitness,
           "size":best_genome.size(),
           "species": best_species_id, "id":best_genome.key}
    print(tmp)
    self.log.append(tmp)
    json.dump(self.log, open("log.json","w"))


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


# env = gym.make("Swimmer-v4")
env = gym.make("Humanoid-v4")

def eval_genome(genome, config, is_render=False):
    global BEST_FITNESS

    observation = env.reset(seed=np.random.randint(0,500))
    manager = reward_manager()

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    current_fitness = 0 

    done = False
    while not done:
        action = net.activate(observation)
        action = np.array(action)/2.5 # range -0.4 to 0.4
        observation, reward, done, info = env.step(action=action)
        current_fitness = manager.step(current_fitness, reward)

    genome.fitness = current_fitness
    env.close()
    return genome.fitness


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat.cfg')
p = neat.Population(config)
stats = neat.StatisticsReporter()
# p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(stats)
p.add_reporter(StdOutReporter_mode(True))

pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

for i in range(NUM_GENERATIONS):
    genome = p.run(pe.evaluate, 1)
    best = stats.best_genome()
    if i %50 == 0:
        with open('best_%d.pkl'%i, 'wb') as output:
            pickle.dump(best, output, 1)