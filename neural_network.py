import random
import math
from ActivationFunction import sigmoid, relu, softmax, tanh, leaky_relu, linear

act_func = {
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
    'tanh': tanh,
    'leaky_relu': leaky_relu,
    'linear': linear
}

class Node:
    def __init__(self, activation_function=None, is_input_node=False, is_bias=False, val=None):
        self.is_input_node = is_input_node
        self.is_bias = is_bias
        self.connections_in = []
        if self.is_input_node:
            self.val = val
        elif self.is_bias:
            self.val = 1
        else:
            self.activation_function = act_func[activation_function]

    def calculate_val(self):
        return sum([connection.output() for connection in self.connections_in])

    def output(self):
        if self.is_input_node or self.is_bias:
            return self.val
        return self.activation_function(self.val)

    def add_connection(self, connection):
        self.connections_in.append(connection)
        self.val = self.calculate_val()

        

class Connection:
    def __init__(self, node_in, node_out, weight):
        self.node_in = node_in
        self.weight = weight
        self.node_out = node_out
        self.enable = True

    def output(self):
        return self.node_in.output() * self.weight


class Layer:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(self.nodes)

class FeedForwardNN:
    def __init__(self, input_layer, output_layer, hidden_layers, connections):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.nodes = input_layer.nodes + [h_layer.nodes for h_layer in hidden_layers] + output_layer.nodes
        self.connections = connections

    def setup_connections(self):
        for connection in self.connections:
            if connection.enable:
                connection.node_out.add_connection(connection)

    def activate(self, inputs):
        for i, input_node in enumerate(self.input_layer.nodes):
            input_node.val = inputs[i]

    def output(self):
        self.setup_connections()
        output = []
        for node in self.output_layer.nodes:
            output.append(node.output())
        output = softmax(output)
        return output

def feedforward_nn(genome):
    input_layer = Layer([Node(is_input_node=True) for i in range(32)])
    hidden_layer_1 = Layer([Node('relu') for i in range(20)])
    hidden_layer_2 = Layer([Node('relu') for i in range(12)])
    output_layer = Layer([Node('sigmoid') for i in range(4)])
    connections = []
    layers = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]
    gene_ind = 0
    for i in range(len(layers) - 1):
        for node_in in layers[i].nodes:
            for node_out in layers[i + 1].nodes:
                connections.append(Connection(node_in, node_out, genome[gene_ind])) #Add connections
                gene_ind += 1
    b1 = Node(is_bias=True)
    b2 = Node(is_bias=True)
    b3 = Node(is_bias=True)
    biases = [b1, b2, b3]
    for i, bias in enumerate(biases):
        for node in layers[i + 1].nodes:
            connections.append(Connection(bias, node, genome[gene_ind]))
            gene_ind += 1
    ffnn = FeedForwardNN(input_layer, output_layer, [hidden_layer_1, hidden_layer_2], connections)
    return ffnn