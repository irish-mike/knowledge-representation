from pathfinder_synth.synth_agents import get_synth_reflex_agent
from pathfinder_synth.synth_enviornment import SynthEnvironment
from pathfinder_synth.synth_graphs import components, synth_graph_ideal_path, synth_graph_wrong_path, \
    synth_graph_dead_end


def run(graph, graph_name):
    print(f"\nRunning agent on {graph_name}")
    env = SynthEnvironment(graph)
    synth_reflex_agent = get_synth_reflex_agent()
    env.add_thing(synth_reflex_agent)
    env.run(20)

run(synth_graph_ideal_path, "Ideal Path")
run(synth_graph_wrong_path, "Wrong Path")

components['fx3'].status = 0  # Turn off 'filter3' to create a (dead end)
run(synth_graph_dead_end, "Dead End Graph")

