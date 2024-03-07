from typing import List, Any, Tuple
from random import randrange
from abc import abstractmethod


class InvalidPropagationInputException(Exception):
    pass


class Neuron:
    @abstractmethod
    def activate(self, inputs: List[float]) -> float:
        ...
    
    @abstractmethod
    def update_weights(self, error: float, alpha=0.0001):
        ...
    

class HeavesideNeuron(Neuron):
    def __init__(self, weights: List[float] = [], default_weight: float = 0.0):
        self.weights = weights
        self.default_weight = float(default_weight)

    def activate(self, inputs: List[float]) -> float:
        sum_input = [float(i) for i in inputs]
        # Extend the weights list to match the length of the inputs
        self.weights = self.weights + [self.default_weight for _ in range(len(sum_input) - len(self.weights))]
        weighted_sum = sum([p[0] * p[1] for p in zip(sum_input, self.weights)])
        return self._heaveside(weighted_sum)

    def _heaveside(self, input: float):
        if input >= 0:
            return 1
        else:
            return 0
    
    def update_weights(self, error: float, alpha=0.01):
        self.weights = [w + error*alpha for w in self.weights]


class NeuronLayer:
    def __init__(self):
        self.neurons: List[Neuron] = []
    
    def add_neruon(self, neuron: Neuron):
        self.neurons.append(neuron)
    
    def size(self):
        return len(self.neurons)

    def activate(self, input: List[Any]) -> List[Any]:
        return [n.activate(input) for n in self.neurons]


# To keep things simple, this brain will fully connect all neurons
# from one layer to the next in a feed forward method.
class Brain:
    def __init__(self):
        self.layers: List[NeuronLayer] = [NeuronLayer()]
    
    def add_neuron(self, neuron: Neuron):
        self.layers[-1].add_neruon(neuron)
        return self
    
    def next_layer(self):
        self.layers.append(NeuronLayer())
        return self

    def propagate(self, input: List[Any]) -> List[Any]:
        output = input
        for layer in self.layers:
            output = layer.activate(output)
        return output
    
    def derivative(self, layer: int, neuron: int) -> str:
        if layer + 1 >= len(self.layers):
            return 1
        retval = 0
        for nrn_num, nrn in enumerate(self.layers[layer+1].neurons):
            if layer + 2 >= len(self.layers):
                retval += nrn.weights[neuron]
            else:
                retval += nrn.weights[neuron] * self.derivative(layer+1, nrn_num)
        return retval
    
    def train(self, data: List[Tuple[List, bool]], alpha: float = 0.01):
        # snapshots = []
        for input, expected in data:
            actual = self.propagate(input)
            error = actual[0]-expected
            
            # pick a random nrn to update
            lyr_num = randrange(0, len(self.layers))
            nrn_num = randrange(0, len(self.layers[lyr_num].neurons))
            error = error*self.derivative(lyr_num, nrn_num)
            nrn = self.layers[lyr_num].neurons[nrn_num]
            nrn.update_weights(error, alpha)
        #     snapshots.append(deepcopy(self))
        # return snapshots
