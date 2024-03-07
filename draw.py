from dash import Dash, html, dcc, callback, Input, Output
import dash_cytoscape as cyto
from brain import Brain, HeavesideNeuron
from random import random
from math import floor
from copy import deepcopy
from typing import List
import plotly.express as px


def func(x):
    return False
    val = (x-1)**3 + (x-1)**2
    return val < 0.04

brain = (
    Brain()
    .add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron())
)
vals = [random() for _ in range(0, 1000)]
dataset = [
    ([x], func(x),)
    for x in vals
]

brains: List[Brain] = []
for _ in range(100):
    brain.train(dataset, alpha=0.001)
    brains.append(deepcopy(brain))

def init_app():
    app = Dash(__name__)

    app.layout = html.Div([
        cyto.Cytoscape(
            id='nrn-network',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '800px'},
            stylesheet=[
                {
                    'selector':'edge[label]',
                    'style':{
                        'label':'data(label)',
                    }
                },
                {
                    'selector':'node[label]',
                    'style':{
                        'label':'data(label)',
                    }
                },
                {
                    'selector': f'[weight >= 1]',
                    'style':{
                        'opacity': 1,
                    }
                }
            ] + [
                {
                    'selector': f'[weight <= {i/10}]',
                    'style':{
                        'opacity': i/10,
                    }
                }
                for i in range(10, 0, -1)
            ],
            # layout={
            #     # 'name': 'cose',
            #     'idealEdgeLength': 100,
            #     'nodeOverlap': 20,
            #     'refresh': 20,
            #     # 'fit': True,
            #     'padding': 30,
            #     'randomize': False,
            #     'componentSpacing': 100,
            #     'nodeRepulsion': 400000,
            #     'edgeElasticity': 100,
            #     'nestingFactor': 5,
            #     'gravity': 80,
            #     'numIter': 1000,
            #     'initialTemp': 200,
            #     'coolingFactor': 0.95,
            #     'minTemp': 1.0
            # },
            responsive=True,
        ),
        dcc.Slider(0, len(brains) - 1, floor(len(brains)/50),
               value=4,
               id='itr-slider'
        ),
        dcc.Graph(id='model-error'),
        dcc.Graph(id='dataset'),
        dcc.Graph(id='inference'),
    ])
    app.run(debug=True)


@callback(
    Output('nrn-network', 'elements'),
    Input('itr-slider', 'value'),
)
def update_output(value):
    brain = brains[value]
    max_lyr_cnt = max([lyr.size() for lyr in brain.layers])
    nodes = [
        {"data": {"id": f"{lyr_num}-{nrn_num}", "label": f"{lyr_num}-{nrn_num}"}, "position": {"x": 100*lyr_num, "y": 100*((max_lyr_cnt-lyr.size())/2+nrn_num) + 1}}
        for lyr_num, lyr in enumerate(brain.layers)
        for nrn_num, nrn in enumerate(lyr.neurons)
    ]
    edges = [
        {"data": {"source":  f"{lyr_num-1}-{prev_nrn_num}", "target":  f"{lyr_num}-{nrn_num}", "label": f"W: {nrn.weights[prev_nrn_num]}", "weight": nrn.weights[prev_nrn_num]}}
        for lyr_num, lyr in enumerate(brain.layers) if lyr_num > 0
        for nrn_num, nrn in enumerate(lyr.neurons)
        for prev_nrn_num, prev_nrn in enumerate(brain.layers[lyr_num-1].neurons)
    ]
    return nodes + edges


@callback(
    Output('model-error', 'figure'),
    Input('itr-slider', 'value'),
)
def update_error(value):
    brain: Brain = brains[value]
    return px.scatter(
        x=[p[0][0] for p in dataset],
        y=[p[1] - brain.propagate(p[0])[0] for p in dataset],
    )


@callback(
    Output('dataset', 'figure'),
    Input('itr-slider', 'value'),
)
def update_error(value):
    return px.scatter(
        x=[p[0][0] for p in dataset],
        y=[p[1] for p in dataset],
    )

@callback(
    Output('inference', 'figure'),
    Input('itr-slider', 'value'),
)
def update_error(value):
    brain: Brain = brains[value]
    return px.scatter(
        x=[p[0][0] for p in dataset],
        y=[brain.propagate(p[0])[0] for p in dataset],
    )
