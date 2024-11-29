from matplotlib import pyplot as plt

from aima.agents import compare_agents
from pathfinder_synth.synth_agents import agent_factory
from pathfinder_synth.synth_environment import synth_environment_factory



def run_agent_comparison():
    # Run the comparison between the agents
    graph_types = ['ideal_path', 'wrong_path', 'dead_end']

    # Print the header for the results table
    print(f"{'Agent Name':<25} {'Environment':<20} {'Average Score':<15}")
    print("-" * 60)

    for graph_type in graph_types:

        agent_factories = [
            agent_factory('Reflex', graph_type),
            agent_factory('Model', graph_type),
            agent_factory('Utility', graph_type),
        ]

        results = compare_agents(lambda: synth_environment_factory(graph_type), agent_factories, n=10, steps=10000)

        # Loop through the results and print each agent's name and average score
        for agent_factory_func, avg_score in results:
            agent_instance = agent_factory_func()
            # Neatly formatted output with fixed-width columns
            print(f"{agent_instance.name:<25} {graph_type:<20} {avg_score:<15.2f}")

# This code has been adapted from the code provided by Ruairí D. O’Reilly's solution for lab 2
def run_agent_comparison_visualise_results():
    graph_types = ['ideal_path', 'wrong_path', 'dead_end']
    agent_names = ['SynthReflexAgent', 'SynthModelBasedAgent', 'SynthUtilityBasedAgent']
    agent_colors = {'SynthReflexAgent': 'r', 'SynthModelBasedAgent': 'b', 'SynthUtilityBasedAgent': 'g'}
    line_styles = {'ideal_path': '-', 'wrong_path': '--', 'dead_end': '-.'}
    markers = {'ideal_path': 'o', 'wrong_path': 's', 'dead_end': '^'}
    step_sizes = [2 ** i for i in range(1, 9)]

    # Initialize performance results dictionary
    performance_results = {
        agent_name: {graph_type: [] for graph_type in graph_types}
        for agent_name in agent_names
    }

    # Run comparisons and collect results
    for graph_type in graph_types:
        # Define agent_factories inside this function
        agent_factories = [
            agent_factory('Reflex', graph_type),
            agent_factory('Model', graph_type),
            agent_factory('Utility', graph_type),
        ]

        for steps in step_sizes:
            results = compare_agents(
                lambda: synth_environment_factory(graph_type),
                agent_factories,
                n=10,
                steps=steps
            )
            for agent_factory_func, avg_score in results:
                agent_name = agent_factory_func().__class__.__name__
                performance_results[agent_name][graph_type].append(avg_score)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for agent_name in agent_names:
        for graph_type in graph_types:
            plt.plot(
                step_sizes,
                performance_results[agent_name][graph_type],
                label=f'{agent_name} ({graph_type})',
                linestyle=line_styles[graph_type],
                marker=markers[graph_type],
                color=agent_colors[agent_name]
            )

    # Plot formatting
    plt.title('Agent Performance Across Different Graphs (Powers of 2)')
    plt.xlabel('Step Size')
    plt.ylabel('Average Performance')
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


run_agent_comparison()

run_agent_comparison_visualise_results()


