from aima.search import UndirectedGraph
from dataclasses import dataclass
import copy

@dataclass
class Component:
    name: str
    status: bool = True  # status is on True or off False
    utility: int = 3

# initialize all components with default on status
def get_components():
    return copy.deepcopy({
        'input': Component('input'),
        'osc1': Component('osc1'),
        'osc2': Component('osc2'),
        'osc3': Component('osc3', status=False),
        'mixer1': Component('mixer1', utility=1),
        'filter1': Component('filter1'),
        'filter2': Component('filter2', status=False),
        'adsr': Component('adsr', utility=1),
        'fx1': Component('fx1'),
        'fx2': Component('fx2'),
        'fx3': Component('fx3'),
        'mixer2': Component('mixer2', utility=1),
        'output': Component('output'),
    })


def graph_factory(graph_type):
    if graph_type == 'ideal_path':
        return UndirectedGraph({
            'input': {'osc1': 10},
            'osc1': {'osc2': 5, 'mixer1': 15},
            'osc2': {'osc3': 5, 'mixer1': 20},
            'osc3': {'mixer1': 5},
            'mixer1': {'filter1': 10},
            'filter1': {'filter2': 5, 'adsr': 10},
            'filter2': {'adsr': 5},
            'adsr': {'fx1': 10},
            'fx1': {'fx2': 5, 'mixer2': 10},
            'fx2': {'fx3': 5, 'mixer2': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5}
        })

    elif graph_type == 'wrong_path':
        return UndirectedGraph({
            'input': {'osc1': 10},
            'osc1': {'mixer1': 5, 'osc2': 10},
            'osc2': {'osc3': 10, 'mixer1': 5},
            'osc3': {'mixer1': 5},
            'mixer1': {'filter1': 10},
            'filter1': {'adsr': 10, 'filter2': 5},
            'filter2': {'adsr': 5},
            'adsr': {'fx1': 10},
            'fx1': {'fx2': 5, 'mixer2': 10},
            'fx2': {'fx3': 5, 'mixer2': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5}
        })

    elif graph_type == 'dead_end':
        return UndirectedGraph({
            'input': {'osc1': 10},
            'osc1': {'osc2': 5, 'mixer1': 15},
            'osc2': {'filter2': 20, 'osc3': 5},
            'osc3': {'mixer1': 5},
            'mixer1': {'filter1': 10},
            'filter1': {'adsr': 10},
            'adsr': {'fx1': 10},
            'fx1': {'fx2': 5, 'mixer2': 5},
            'fx2': {'fx3': 5, },
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5}
        })

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
