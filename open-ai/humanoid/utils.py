import json
from neat.math_util import mean, stdev
import neat
import matplotlib.pyplot as plt
import warnings
import numpy as np

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
           "Best_fitness":best_genome.fitness,
           "size":best_genome.size(),
           "species": best_species_id, "id":best_genome.key}
    print(tmp)
    self.log.append(tmp)
    json.dump(self.log, open("log.json","w"))


class reward_manager():
    def __init__(self, factor=0.98):
        self.num_step = 1
        self.factor = factor

    def reset(self ):
        self.__init__()

    def step(self, value, reward):
        value = value + (self.factor**self.num_step)*reward
        self.num_step+=1
        return value

def plot_stats(path, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return
    statistics = json.load(open(path))
    generation = range(len(statistics))
    best_fitness = np.array([c["Best_fitness"] for c in statistics])
    avg_fitness = np.array([c["Pop_avg_fitness"] for c in statistics])
    stdev_fitness = np.array([c["stdev"] for c in statistics])

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()