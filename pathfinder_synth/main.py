from pathfinder_synth.synth_agents import get_synth_reflex_agent, get_synth_model_based_agent
from pathfinder_synth.synth_enviornment import SynthEnvironment
from pathfinder_synth.synth_graphs import components, synth_graph_ideal_path, synth_graph_wrong_path, \
    synth_graph_dead_end


def run(graph, agent, graph_name):
    print(f"\nRunning {agent.__name__} on {graph_name}")
    env = SynthEnvironment(graph)
    env.add_thing(agent)
    env.run(20)


# Run Reflex Agent
run(synth_graph_ideal_path, get_synth_reflex_agent(), "Ideal Path")
run(synth_graph_wrong_path, get_synth_reflex_agent(), "Wrong Path")

components['fx3'].status = 0  # Turn off 'filter3' to create a (dead end)
run(synth_graph_dead_end, get_synth_reflex_agent(), "Dead End Graph")

# Run Model based Agent
run(synth_graph_ideal_path, get_synth_model_based_agent(), "Ideal Path")
run(synth_graph_wrong_path, get_synth_model_based_agent(), "Wrong Path")
run(synth_graph_dead_end, get_synth_model_based_agent(), "Dead End Graph")