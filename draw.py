from dash import Dash, html, dcc, callback, Input, Output, State
import dash_cytoscape as cyto
from brain import Brain, HeavesideNeuron
from random import random
from math import ceil
from copy import deepcopy
from typing import List
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd


def func(*x):
    return x[0] ^ x[1]
    # return False
    return x < 0.5
    val = (x-1)**3 + (x-1)**2
    return val >= 0.04

brain = (
    Brain()
    .add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron()).add_neuron(HeavesideNeuron())
    .next_layer().add_neuron(HeavesideNeuron())
)
# brain = (
#     Brain()
#     .add_neuron(HeavesideNeuron())
#     .next_layer().add_neuron(HeavesideNeuron())
# )
# vals = [random() for _ in range(0, 100)]
# dataset = [
#     ([x], func(x),)
#     for x in vals
# ]

dataset = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 0),
]

brains: List[Brain] = []
brain_error: List[int] = []

def network_graph(id, style):
    return cyto.Cytoscape(
        id=id,
        layout={'name': 'preset'},
        style=style,
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
        ] + [
            {
                'selector': f'[weight < 0.001]',
                'style':{
                    'opacity': 0,
                }
            }
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
    )

def init_app():
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            network_graph(id='nrn-network', style={'height': '800px', 'flex': 2}),
            html.Div([
                dcc.Graph(id='model-error', style={'flex': 1}),
                dcc.Graph(id='dataset', style={'flex': 1}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'flex': 2}),
            html.Div([
                dcc.Input(id="num-rounds"),
                html.Button('Train', id='train', n_clicks=0),
                html.Button('Reset', id='reset', n_clicks=0),
            ], style={'display': 'flex', 'flexDirection': 'column', 'flex': 1}),
        ], style={"width": "100%", 'display': 'flex', 'flexDirection': 'row', 'flex': 1}),
        html.Div([
            dcc.Slider(0, len(brains) - 1, ceil(len(brains)/50), value=4, id='itr-slider'),
        ], style={"width": "100%", 'flex': 1}),
    ], style={'display': 'flex', 'flexDirection': 'column', "align-items": "flex-end"})
    app.run(debug=True)


@callback(
    Output('nrn-network', 'elements'),
    Input('itr-slider', 'value'),
    Input('nrn-network', 'tapNodeData'),
)
def update_output(value, tap_node_data):
    brain = brains[value]
    selected_node = tap_node_data.get('id', "") if tap_node_data else ""
    max_lyr_cnt = max([lyr.size() for lyr in brain.layers])
    nodes = [
        {"data": {"id": f"{lyr_num}-{nrn_num}", "label": f"{lyr_num}-{nrn_num}"}, "position": {"x": 100*lyr_num, "y": 100*((max_lyr_cnt-lyr.size())/2+nrn_num) + 1}}
        for lyr_num, lyr in enumerate(brain.layers)
        for nrn_num, nrn in enumerate(lyr.neurons)
    ]
    edges = [
        (
            {"data": {"source":  f"{lyr_num-1}-{prev_nrn_num}", "target":  f"{lyr_num}-{nrn_num}", "label": f"W: {nrn.weights[prev_nrn_num]:3.3f}", "weight": nrn.weights[prev_nrn_num]}}
            if f"{lyr_num}-{nrn_num}" == selected_node else
            {"data": {"source":  f"{lyr_num-1}-{prev_nrn_num}", "target":  f"{lyr_num}-{nrn_num}", "weight": nrn.weights[prev_nrn_num]}}
        )
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
    # return px.scatter(
    #     x=["".join(str(v) for v in p[0]) for p in dataset],
    #     y=[brain.propagate(p[0])[0] - p[1] for p in dataset],
    #     title="Model Error",
    # )
    return px.scatter(
        x=[i for i in range(len(brain_error))],
        y=brain_error,
        title="Model Error",
    )


@callback(
    Output('dataset', 'figure'),
    Input('itr-slider', 'value'),
)
def update_dataset(value):
    brain: Brain = brains[value]
    data = pd.DataFrame({
        "x": ["".join(str(v) for v in p[0]) for p in dataset] * 2,
        "y": [float(p[1]) for p in dataset] + [float(brain.propagate(p[0])[0]) for p in dataset],
        "type": ["actual"]*len(dataset) + ["predicted"]*len(dataset),
    })
    return px.scatter(data, x="x", y="y", color="type")


@callback(
    Output('itr-slider', 'value'),
    Input('train', 'n_clicks'),
    State('num-rounds', 'value'),
    prevent_initial_call=True,
)
def retrain(_, num_rounds):
    global brains
    for _ in range(int(num_rounds)):
        brain.train(dataset, alpha=0.01)
        brains.append(deepcopy(brain))
        brain_error.append(brain.error(dataset))
        brains.pop(0)
    return 0


@callback(
    Output('itr-slider', 'max'),
    Input('reset', 'n_clicks'),
    State('num-rounds', 'value'),
    prevent_initial_call=True,
)
def reset(_, num_rounds):
    num_rounds = int(num_rounds)
    global brains
    brains.clear()
    brain_error.clear()
    for _ in range(num_rounds):
        brain.train(dataset, alpha=0.01)
        brains.append(deepcopy(brain))
        brain_error.append(brain.error(dataset))
    return num_rounds-1
