import random
from copy import deepcopy, copy
from neural_network import feedforward_nn

class Individual:
    def __init__(self, length, num_weights, weights_range, bias_range):
        self.length = length
        self.num_weights = num_weights
        self.num_biases = self.length - self.num_weights
        self.weights = [round(random.uniform(weights_range[0], weights_range[1]), 2) for i in range(num_weights)]
        self.biases = [round(random.uniform(bias_range[0], bias_range[1]), 2) for i in range(self.length - num_weights)]
        self.genome = self.weights + self.biases
        self.fitness = 0


class Population:
    def __init__(self, pop_size, ind_length, ind_num_weights, ind_weights_range, ind_bias_range, num_gen):
        self.pop_size = pop_size
        self.ind_length = ind_length
        self.ind_num_weights = ind_num_weights
        self.ind_weights_range = ind_weights_range
        self.ind_bias_range = ind_bias_range
        self.num_gen = num_gen
        self.current_gen = 0
        self.pop = [Individual(self.ind_length, self.ind_num_weights, self.ind_weights_range, self.ind_bias_range) for i in range(self.pop_size)]



def mutate(individual, weight_mut_prob, bias_mut_prob, weight_mut_std, bias_mut_std):
    offspring = deepcopy(individual)
    for gene in offspring.weights:
        if random.random() < weight_mut_prob:
            gene = random.uniform(gene - weight_mut_std, gene + weight_mut_std)
    for gene in offspring.biases:
        if random.random() < bias_mut_prob:
            gene = random.uniform(gene - bias_mut_std, bias_mut_std)
    return offspring

def crossover(individual1, individual2, cx_prob):
    offspring1 = deepcopy(individual1)
    offspring2 = deepcopy(individual2)
    for i in range(individual1.length):
        if random.random() < cx_prob:
            temp = offspring1.genome[i]
            offspring1.genome[i] = offspring2.genome[i]
            offspring2.genome[i] = temp
    return offspring1, offspring2

