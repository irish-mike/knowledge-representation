from aima.search import UndirectedGraph
from dataclasses import dataclass

@dataclass
class Component:
    name: str
    status: int = 1  # status is on (1) or off (0)

# initialize all components with default on status
components = {
    'input': Component('input'),
    'osc1': Component('osc1'),
    'osc2': Component('osc2'),
    'osc3': Component('osc3'),
    'mixer1': Component('mixer1'),
    'filter1': Component('filter1'),
    'filter2': Component('filter2'),
    'adsr': Component('adsr'),
    'fx1': Component('fx1'),
    'fx2': Component('fx2'),
    'fx3': Component('fx3'),
    'mixer2': Component('mixer2'),
    'output': Component('output'),
}

# This is the ideal graph layout, the simple agent will follow the correct path because all the components are in the correct order.
synth_graph_ideal_path = UndirectedGraph({
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

# In this graph the components are not in sequential order, so a simple agent will take the wrong path
synth_graph_wrong_path = UndirectedGraph({
    'input': {'osc1': 10},
    'osc1': {'mixer1': 5, 'osc2': 10},
    'osc2': {'mixer1': 5, 'osc3': 10},
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

# This graph contains a dead end, an agent without knowledge of its environment will get stuck here
synth_graph_dead_end = UndirectedGraph({
    'input': {'osc1': 10},
    'osc1': {'osc2': 5, 'mixer1': 15},
    'osc2': {'filter2': 20, 'osc3': 5},
    'osc3': {'mixer1': 5},
    'mixer1': {'filter1': 10, 'lfo2': 25},
    'filter1': {'adsr': 10},
    'adsr': {'fx1': 10},
    'fx1': {'fx2': 5, 'mixer2': 5},
    'fx2': {'fx3': 5,},
    'fx3': {'mixer2': 10},
    'mixer2': {'output': 5}
})
