from typing import List, Any, Tuple


class InvalidPropagationInputException(Exception):
    pass


class Neuron:
    def activate(self, inputs: List[float]) -> float:
        raise NotImplementedError("Neuron `activate` function has not been implemented!")
    
    def update_weights(self, error: float, prev_lyr: List[float], alpha=0.0001):
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
    
    def update_weights(self, error: float, prev_lyr: List[float], alpha=0.0001):
        for i, weight in enumerate(self.weights):
            self.weights[i] = self.weights - (error * weight * prev_lyr[i] * alpha)


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
    
    def train(self, data: List[Tuple[List, bool]]):
        for input, expected in data:
            layer_outputs = []
            for i, layer in enumerate[self.layers]:
                layer_outputs[i] = layer.activate(input if i == 0 else layer_outputs[i-1])
            actual = self.propagate(input)
            error = actual-expected
            for lyr_num, layer in reversed(list(enumerate(self.layers))):
                if lyr_num == 0:
                    continue
                for nrn in layer.neurons:
                    nrn.update_weights(error, layer_outputs[lyr_num-1])
                
