import numpy as np
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(x, 0)

def softmax(x : list):
    exp_list = [math.exp(ele) for ele in x]
    return [exp_ele / sum(exp_list) for exp_ele in exp_list]

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def linear(x):
    return x