import pytest
from brain import Brain, HeavesideNeuron

def test_derivative():
    b = (Brain()
         .add_neuron(HeavesideNeuron(weights=[0.1]))
         .next_layer().add_neuron(HeavesideNeuron(weights=[0.2])).add_neuron(HeavesideNeuron(weights=[0.3]))
         .next_layer().add_neuron(HeavesideNeuron(weights=[0.4, 0.5])).add_neuron(HeavesideNeuron(weights=[0.6,0.7])).add_neuron(HeavesideNeuron(weights=[0.8,0.9]))
         .next_layer().add_neuron(HeavesideNeuron(weights=[0.11, 0.12, 0.13]))
    )
    assert b.derivative(0, 0) == pytest.approx(0.1208)
    assert b.derivative(0, 0) == 0.2*b.derivative(1, 0) + 0.3*b.derivative(1, 1)
