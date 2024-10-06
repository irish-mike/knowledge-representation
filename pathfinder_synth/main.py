from aima.search import Graph
from pathfinder_synth.simple_agent import agent as simple_agent
from pathfinder_synth.synth_enviornment import SynthEnvironment

# Basic Synth graph
synth_graph = Graph({
    'input': {'osc1': 1},
    'osc1': {'osc2': 1},
    'osc2': {'mixer1': 1},
    'mixer1': {'filter1': 1},
    'filter1': {'adsr': 1},
    'adsr': {'fx1': 1},
    'fx1': {'mixer2': 1},
    'mixer2': {'output': 1},
    'output': {}
}, directed=False)

env = SynthEnvironment(synth_graph)
env.add_thing(simple_agent)
env.run()
