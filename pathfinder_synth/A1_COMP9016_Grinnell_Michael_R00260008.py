import copy
import time
from dataclasses import dataclass
from enum import Enum
from random import choice

import numpy as np
from matplotlib import pyplot as plt

from agents import Environment, Agent, compare_agents
from search import UndirectedGraph, GraphProblem, depth_limited_search, astar_search, \
    greedy_best_first_graph_search, recursive_best_first_search, InstrumentedProblem, breadth_first_graph_search, \
    depth_first_graph_search
from utils import print_table


# Defines a component node and its properties
@dataclass
class Component:
    name: str
    status: bool = True  # status is on True or off False
    signal: bool = False # has a signal to process

# Define the Actions an agent can do
class Action(Enum):
    PROCESS = 'process'
    MOVE = 'move'
    STUCK = 'stuck'


"""_____________________ ENVIRONMENT  _____________________"""

class SynthEnvironment(Environment):
    def __init__(self, graph, components):
        super().__init__()
        self.components = components
        self.graph = graph

    def percept(self, agent):

        location = agent.location

        neighbors = self.graph.get(location)
        neighbor_nodes = list(neighbors.keys()) if neighbors else []

        signal = self.components[agent.location].signal

        return location, signal, neighbor_nodes

    def execute_action(self, agent, action):

        key, value = action

        if key == "move":
            self.handle_move(agent, value)
        elif key == "process":
            self.handle_process(agent, value)
        else:
            self.handle_invalid_action(agent, action)

        agent.performance -= 1  # any action the agent takes incurs a penalty

    def handle_move(self, agent, component):

        # If the component is off, the environment won't allow the agent to move
        if not self.components[component].status:
            agent.performance -= 5  # Penalty for trying to make an invalid move
            return

        agent.location = component

    def handle_process(self, agent, value):
        if self.components[value].signal:
            self.components[value].signal = False # simulate processing a signal
            agent.performance += 5

    def handle_invalid_action(self, agent, action):
        # Agent has become self-aware and is trying to do actions outside the defined scope, kill it.
        agent.alive = False
        agent.performance -= 100

    def get_random_neighbor(self, location):
        neighbors = self.graph.get(location)
        return choice(list(neighbors.keys()))

    def get_random_parent(self, component):
        parents = []
        for node, neighbors in self.graph.items():
            if component in neighbors:
                parents.append(node)

        return choice(parents) if parents else None

    def is_done(self):
        for agent in self.agents:
            self.handle_goal_state(agent)

        return super().is_done()

    def handle_goal_state(self, agent):
        # agent has reached the output and completed the goal
        if agent.location == "output":
            agent.performance += 10
            agent.alive = False

    def default_location(self, thing):
        return "input"

"""_____________________ AGENT CLASSES _____________________"""

'''
A simple reflex agent that makes decisions based solely on the current percept.

This agent has a hard time navigating in the synth environment.
Since it only knows about its current state, it cannot make informed decisions about where to move next.

Therefore its movement is random
'''
class SynthReflexAgent(Agent):

    def __init__(self, name):
        super().__init__(self.program)
        self.name = name

    def program(self, percept):
        location, signal, possible_moves = percept

        if signal:
            return Action.PROCESS.value, location

        if not possible_moves:
            return Action.STUCK.value, location

        return Action.MOVE.value, choice(possible_moves)


'''
A model-based agent that keeps track of visited components to avoid revisiting nodes and loops.
It can navigate better than the reflex agent since it knows where it has already been.
'''
class SynthModelBasedAgent(SynthReflexAgent):
    START_COMPONENT = 'input'

    def __init__(self, name):
        super().__init__(name)
        self.previous_components = [self.START_COMPONENT]  # Initialize with starting component
        self.inactive_components = []

    def get_unvisited(self, possible_moves):
        # Returns the difference between the possible moves and previous nodes.
        return [move for move in possible_moves if move not in self.previous_components]

    def has_not_moved(self, location):
        return location == self.previous_components[-1]

    def backtrack(self, location):
        """
        We need to check if the agent is stuck,
        somtimes it keeps revisiting the same components because the environment rejects its attempt to move to an off component
        So it keeps trying to backtrack to the off component, to do this we just check if the component the agent is trying to move to is in our
        visited multiple times, in this case 3.
        """

        current_index = self.previous_components.index(location)
        next_location = self.previous_components[current_index - 1] if current_index > 0 else None

        if self.previous_components.count(next_location) >= 3:
            first_index = self.previous_components.index(next_location)
            return self.previous_components[first_index - 1] if first_index > 0 else None

        return next_location

    def program(self, percept):
        location, signal, possible_moves = percept

        if signal:
            return 'process', location

        """
            If there are unvisited moves, choose randomly.
            If no unvisited moves are available, backtrack to the previous node.
            otherwise we are stuck
        """

        unvisited_moves = self.get_unvisited(possible_moves)

        if unvisited_moves:
            next_location = choice(unvisited_moves)
        else:
            next_location = self.backtrack(location)

        if next_location:
            self.previous_components.append(next_location)
            return Action.MOVE.value, next_location
        else:
            return Action.STUCK.value, location


'''
This Utility based agent performs best in the synth environment, 
it can keep track of where it has already been just like the reflex agent.
The advantage it has is that it will pick the next best moved based on what components are on and have a signal to process.
'''
class SynthUtilityBasedAgent(SynthModelBasedAgent):
    def __init__(self, name, components):
        super().__init__(name)
        self.components = components

    def filter_active_components(self, moves):
        """
         Filters and returns components that are active (status is True).
        """
        return [
            comp_name for comp_name in moves
            if comp_name in self.components and self.components[comp_name].status
        ]

    def get_highest_utility(self, moves):
        """
        Returns the move with the highest utility,
        for this implementation it is any component with a signal
        """
        if not moves:
            return None
        return max(
            moves,
            key=lambda comp_name: self.components[comp_name].signal,
            default=None
        )

    def program(self, percept):
        location, signal, possible_moves = percept

        if signal:
            return Action.PROCESS.value, location

        # Get the components that have not been visited
        unvisited = self.get_unvisited(possible_moves)

        # Exclude components that are inactive
        active_components = self.filter_active_components(unvisited)

        best = self.get_highest_utility(active_components)

        if not best:
            best = self.backtrack(location)

        if best:
            self.previous_components.append(best)
            return Action.MOVE.value, best
        else:
            return Action.STUCK.value, location

"""_____________________ PROBLEM  _____________________"""

class SynthGraphProblem(GraphProblem):

    def __init__(self, initial, goal, graph, components):
        super().__init__(initial, goal, graph)
        self.components = components

    def actions(self, location):
        actions = []

        if self.components[location].signal:
            actions.append(('process', location))
            self.components[location].signal = False

        available_moves = super().actions(location)

        # Filter the valid moves (only components that are 'on')
        valid_moves = self.get_valid_moves(available_moves)

        # If we want the search to traverse the nodes with a signal first, this is where we could do it
        # just filter the nodes based on ones with a signal until there are none left.

        # Add 'move' action as key
        for move in valid_moves:
            actions.append(('move', move))

        return actions

    def get_valid_moves(self, moves):
        return [move for move in moves if self.components[move].status]

    def result(self, state, action):
        action_type, target = action

        if action_type == 'move':
            return target

        return state  # If processing

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, cost_so_far, A, action, B):
        action_type, target = action
        if action_type == 'move':
            return super().path_cost(cost_so_far, A, target, B)
        else:
            return cost_so_far + 1  # +1 for signal processing

    def h(self, node):
        # The idea here is to choose the path with the least latency
        min_edge_cost = self.find_min_edge()
        remaining_steps_estimate = 1  # Fixed estimate for remaining steps, should probably implement something a bit more involved

        return min_edge_cost * remaining_steps_estimate


"""_____________________ FACTORIES  _____________________"""

def component_factory(graph_type):
    components = {
        'input': Component('input'),
        'osc1': Component('osc1', signal=True),
        'osc2': Component('osc2', signal=True),
        'osc3': Component('osc3', status=False),
        'osc4': Component('osc4', signal=True),
        'osc5': Component('osc5', signal=True),
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
        components['osc5'].signal = components['osc5'].status = False
        components['fx3'].signal = components['fx3'].status = True

    return copy.deepcopy(components)

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
            'input': {'osc1': 10, 'osc2': 5, 'osc3': 5},
            'osc1': {'mixer1': 5, 'osc2': 10},
            'osc2': {'osc3': 10, 'mixer1': 5},
            'osc3': {'osc4': 5, 'osc5': 10, 'mixer1': 10},
            'osc4': {'osc5': 10},
            'osc5': {'mixer1': 5},
            'mixer1': {'filter1': 10, 'filter2': 15, 'filter3': 20},
            'filter1': {'adsr': 10, 'filter2': 20},
            'filter2': {'filter3': 5, 'adsr': 5},
            'filter3': {'adsr': 10,},
            'adsr': {'fx1': 10, 'fx2': 5, 'fx3': 10},
            'fx1': {'fx2': 5, 'mixer2': 10},
            'fx2': {'fx3': 5, 'mixer2': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5},
        })

    elif graph_type == 'dead_end':
        return UndirectedGraph({
            'input': {'osc1': 10, 'osc2': 5, 'osc3': 5, 'osc4': 5},
            'osc1': {'mixer1': 5, 'osc2': 10},
            'osc2': {'osc3': 10, 'mixer1': 5},
            'osc3': {'osc4': 5, 'osc5': 10},
            'osc4': {'osc5': 10},
            'osc5': {'mixer1': 5},
            'mixer1': {'filter1': 10, 'filter2': 15, 'filter3': 20},
            'filter1': {'adsr': 10, 'filter2': 20},
            'filter2': {'filter3': 5, 'adsr': 5},
            'filter3': {'adsr': 10,},
            'adsr': {'fx1': 10, 'fx2': 5, 'fx3': 10},
            'fx1': {'fx2': 5, 'mixer2': 10},
            'fx2': {'fx3': 5, 'mixer2': 5},
            'fx3': {'mixer2': 10},
            'mixer2': {'output': 5},
        })

def synth_environment_factory(graph_type):
    graph = graph_factory(graph_type)
    components = component_factory(graph_type)
    return SynthEnvironment(graph, components)

def agent_factory(agent_type, graph_type):
    def create_agent():
        if agent_type == 'Reflex':
            return SynthReflexAgent(agent_type)
        elif agent_type == 'Model':
            return SynthModelBasedAgent(agent_type)
        elif agent_type == 'Utility':
            return SynthUtilityBasedAgent(agent_type, component_factory(graph_type))

    return create_agent

def depth_limited_search_factory(p):
    return depth_limited_search(p, limit=7)

def astar_search_factory(p):
    return astar_search(p, h=p.h)

def greedy_best_first_graph_search_factory(p):
    return greedy_best_first_graph_search(p, f=p.h)

def recursive_best_first_search_search_factory(p):
    return recursive_best_first_search(p, h=p.h)

"""_____________________ ANALYSIS FUNCTIONS  _____________________"""

def run_agent_comparison():
    graph_types = ['ideal_path', 'wrong_path', 'dead_end']

    print(f"{'Agent Name':<25} {'Environment':<20} {'Average Score':<15}")
    print("-" * 60)

    for graph_type in graph_types:

        agent_factories = [
            agent_factory('Reflex', graph_type),
            agent_factory('Model', graph_type),
            agent_factory('Utility', graph_type),
        ]

        results = compare_agents(lambda: synth_environment_factory(graph_type), agent_factories, n=10, steps=10000)

        for agent_factory_func, avg_score in results:
            agent_instance = agent_factory_func()
            print(f"{agent_instance.name:<25} {graph_type:<20} {avg_score:<15.2f}")

# The function below has been adapted from the code provided by Ruairí D. O’Reilly's solution for lab 2
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

def average_search_time(problem, search_function, runs):
    total_time = 0

    for _ in range(runs):
        start_time = time.time()
        search_function(problem)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / runs

    return average_time

def compare_graph_searchers(searchers, graph_types, runs=10):
    problems = []
    headers = ['Searcher', 'Graph Type', 'Average Nodes Expanded', 'Solution Length', 'Path Cost', 'Average Time (s)']

    for graph_type in graph_types:
        graph = graph_factory(graph_type)
        components = component_factory(graph_type)
        problem = SynthGraphProblem('input', 'output', graph, components)
        problems.append((graph_type, problem))

    results = []

    for searcher in searchers:
        searcher_name = searcher.__name__

        for graph_type, problem in problems:

            inst_problem = InstrumentedProblem(problem)
            avg_time = average_search_time(inst_problem, searcher, runs=runs)

            search_result = searcher(inst_problem)

            # Collect metrics: nodes expanded, solution length, path cost
            nodes_expanded = int(inst_problem.succs / runs)  # Average Number of nodes expanded
            solution_length = len(search_result.solution()) if search_result else 'N/A'
            path_cost = search_result.path_cost if search_result else 'N/A'

            results.append([searcher_name, graph_type, nodes_expanded, solution_length, path_cost, f"{avg_time:.6f} seconds"])

    print_table(results, headers)


"""_____________________ RUNNING THE CODE  _____________________"""

print(f"\nRunning Agent comparison...\n")
run_agent_comparison()

print(f"\nGenerating Graph Visualization...\n")
run_agent_comparison_visualise_results()


# Search algorithms to compare
searchers = [
    breadth_first_graph_search,
    depth_first_graph_search,
    depth_limited_search_factory,
    greedy_best_first_graph_search_factory,
    recursive_best_first_search_search_factory,
    astar_search_factory,
]

# Graph types to compare
graph_types = ['ideal_path', 'wrong_path', 'dead_end']


print(f"\nComparing Search Algorithms, this might take a moment...\n")
compare_graph_searchers(searchers, graph_types)

