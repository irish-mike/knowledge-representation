from fontTools.misc.cython import returns

from aima.search import UndirectedGraph
from dataclasses import dataclass
import copy

@dataclass
class Component:
    name: str
    status: bool = True  # status is on True or off False
    signal: bool = False # has a signal to process

# initialize all components with default on status
def get_components(graph_type):
    components = {
        'input': Component('input'),
        'osc1': Component('osc1', signal=True),
        'osc2': Component('osc2', signal=True),
        'osc3': Component('osc3', status=False),
        'osc4': Component('osc4', signal=True),
        'osc1': Component('osc1', signal=True),
        'mixer1': Component('mixer1'),
        'filter1': Component('filter1', signal=True),
        'filter2': Component('filter2', status=False),
        'filter3': Component('filter3', signal=True),
        'adsr': Component('adsr', signal=True),
        'fx1': Component('fx1', signal=True),
        'fx2': Component('fx2', status=False),
        'fx3': Component('fx3', status=False),
        'mixer2': Component('mixer2'),
        'output': Component('output'),
    }

    # Turn on all components, so there is many paths.
    if graph_type == 'wrong_path':
        for name in ['osc3', 'filter2', 'fx2', 'fx3']:
            components[name].signal = True
            components[name].status = True

    if graph_type == 'dead_end':
        components['osc3'].signal = components['osc3'].status = True
        components['osc1'].signal = components['osc1'].status = False
        components['fx3'].signal = components['fx3'].status = True

    return copy.deepcopy(components)

def graph_factory(graph_type):
    if graph_type == 'ideal_path':
        return UndirectedGraph({
            'input': {'osc1': 1},
            'osc1': {'osc2': 3, 'mixer1': 2},
            'osc2': {'osc3': 3, 'mixer1': 2},
            'osc3': {'mixer1': 2},
            'mixer1': {'filter1': 2},
            'filter1': {'filter2': 2, 'adsr': 2},
            'filter2': {'adsr': 3},
            'adsr': {'fx1': 2},
            'fx1': {'fx2': 3, 'mixer2': 1},
            'fx2': {'fx3': 3, 'mixer2': 1},
            'fx3': {'mixer2': 1},
            'mixer2': {'output': 1}
        })

    elif graph_type == 'wrong_path':
        return UndirectedGraph({
            'input': {'osc1': 2, 'osc2': 1, 'osc3': 3},
            'osc1': {'mixer1': 1, 'osc2': 2},
            'osc2': {'osc3': 2, 'mixer1': 1},
            'osc3': {'osc4': 2, 'osc1': 2, 'mixer1': 1},
            'osc4': {'osc1': 2},
            'osc1': {'mixer1': 1},
            'mixer1': {'filter1': 2, 'filter2': 1, 'filter3': 4},
            'filter1': {'adsr': 1, 'filter2': 2},
            'filter2': {'filter3': 2, 'adsr': 1},
            'filter3': {'adsr': 1,},
            'adsr': {'fx1': 2, 'fx2': 1, 'fx3': 2},
            'fx1': {'fx2': 2, 'mixer2': 1},
            'fx2': {'fx3': 2, 'mixer2': 1},
            'fx3': {'mixer2': 1},
            'mixer2': {'output': 1},
        })

    elif graph_type == 'dead_end':
        return UndirectedGraph({
            'input': {'osc1': 2, 'osc2': 1, 'osc3': 3, 'osc4': 3},
            'osc1': {'mixer1': 1, 'osc2': 2},
            'osc2': {'osc3': 2, 'mixer1': 1},
            'osc3': {'osc4': 2, 'osc1': 2},
            'osc4': {'osc1': 2},
            'osc1': {'mixer1': 1},
            'mixer1': {'filter1': 2, 'filter2': 1, 'filter3': 4},
            'filter1': {'adsr': 1, 'filter2': 4},
            'filter2': {'filter3': 3, 'adsr': 1},
            'filter3': {'adsr': 2,},
            'adsr': {'fx1': 2, 'fx2': 1, 'fx3': 2},
            'fx1': {'fx2': 3, 'mixer2': 1},
            'fx2': {'fx3': 3, 'mixer2': 1},
            'fx3': {'mixer2': 1},
            'mixer2': {'output': 1},
        })


