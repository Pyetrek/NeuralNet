from dash import Dash, html, dcc, callback, Input, Output
import dash_cytoscape as cyto
from brain import Brain, HeavesideNeuron


def func(x):
    val = (x-1)**3 + (x-1)**2
    return val == 0

brains = [
 Brain()
   .add_neuron(HeavesideNeuron(default_weight=i/10)).add_neuron(HeavesideNeuron(default_weight=i/10))
   .next_layer().add_neuron(HeavesideNeuron(default_weight=i/10)).add_neuron(HeavesideNeuron(default_weight=i/10)).add_neuron(HeavesideNeuron(default_weight=i/10)).add_neuron(HeavesideNeuron(default_weight=i/10))
   .next_layer().add_neuron(HeavesideNeuron(default_weight=i/10))
 for i in range(1, 10)
]

for b in brains:
    b.propagate([1,0])


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
            ] + [
                {
                    'selector': f'[weight <= {i/10}]',
                    'style':{
                        'opacity': i/10,
                    }
                }
                for i in range(10, 1, -1)
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
        dcc.Slider(0, len(brains) - 1, 1,
               value=4,
               id='itr-slider'
        ),
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
