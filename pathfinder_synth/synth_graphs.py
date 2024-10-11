from aima.search import UndirectedGraph
from dataclasses import dataclass
import copy

@dataclass
class Component:
    name: str
    status: bool = False  # status is on True or off False
    signal: bool = False # has a signal to process

# initialize all components with default on status
def get_components():
    return copy.deepcopy({
        'input': Component('input', status=True, signal=False),
        'osc1': Component('osc1', status=True, signal=True),
        'osc2': Component('osc2', status=True, signal=True),
        'osc3': Component('osc3', status=True, signal=True),
        'osc4': Component('osc4', status=False, signal=False),  # Off component, dead end
        'osc5': Component('osc5', status=False, signal=False),  # Off component, dead end
        'mixer1': Component('mixer1', status=True, signal=False),
        'filter1': Component('filter1', status=True, signal=True),
        'filter2': Component('filter2', status=False, signal=False),  # Off component, dead end
        'filter3': Component('filter3', status=False, signal=False),  # Off component, dead end
        'adsr': Component('adsr', status=True, signal=True),
        'fx1': Component('fx1', status=True, signal=True),
        'fx2': Component('fx2', status=True, signal=True),
        'fx3': Component('fx3', status=True, signal=True),
        'mixer2': Component('mixer2', status=True, signal=False),
        'output': Component('output', status=True, signal=False),
        'lfo1': Component('lfo1', status=False, signal=False),  # Off component, dead end
        'lfo2': Component('lfo2', status=False, signal=False)   # Off component, dead end
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
            'osc1': {'mixer1': 5, 'osc2': 10, 'osc3': 20},
            'osc2': {'osc3': 10, 'osc5': 15, 'mixer1': 5},
            'osc3': {'mixer1': 5},
            'osc4': {'osc5': 10},
            'mixer1': {'filter1': 10, 'filter2': 15},
            'filter1': {'adsr': 10, 'filter3': 20},
            'filter2': {'adsr': 5},
            'adsr': {'fx1': 10},
            'fx1': {'fx2': 5, 'mixer2': 10},
            'fx2': {'fx3': 5, 'mixer2': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5},
        })

    elif graph_type == 'dead_end':
        return UndirectedGraph({
            'input': {'osc1': 10},
            'osc1': {'osc2': 5, 'mixer1': 15, 'lfo1': 20},
            'osc2': {'filter2': 20, 'osc3': 5, 'lfo2': 15},
            'osc3': {'mixer1': 5},
            'mixer1': {'filter1': 10},
            'filter1': {'adsr': 10},
            'adsr': {'fx1': 10},
            'fx1': {'fx2': 5, 'mixer2': 5},
            'fx2': {'fx3': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5},
            'lfo1': {'lfo2': 5},
            'lfo2': {'filter2': 15},
            'filter2': {}  # Dead end
        })
