import numpy as np
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

    def output(self):
        output = []
        for node in self.output_layer.nodes:
            output.append(node.output())
        return output

def construct_nn(num_inputs, num_outputs, num_hidden_layers, num_nodes_layers):
    for i in range(num_inputs):
        


if __name__ == "__main__":
    x = Node('sigmoid', is_input_node=True, val=0)
    y = Node('sigmoid', is_input_node=True, val=0)
    input_layer = Layer([x, y])
    z = Node('sigmoid')
    t = Node('sigmoid')
    hidden_layer = Layer([z, t])
    o = Node('sigmoid')
    output_layer = Layer([o])

    b1 = Node(is_bias=True)
    b2 = Node(is_bias=True)

    c1 = Connection(x, z, 8.)
    c2 = Connection(x, t, 6.73)
    c3 = Connection(y, z, -7.09) 
    c4 = Connection(y, t, -6.07)   
    c5 = Connection(b1, z, -4.1)
    c6 = Connection(b1, t, 2.94)
    c7 = Connection(z, o , 8.)
    c8 = Connection(t, o, -8.)
    c9 = Connection(b2, o, 3.78)

    connections = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
    
    ffnn = FeedForwardNN(input_layer, output_layer, [hidden_layer], connections)
    ffnn.setup_connections()
    print(ffnn.output())