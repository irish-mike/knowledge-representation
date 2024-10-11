import sys
import numpy as np
from aima.search import (
    GraphProblem,
    breadth_first_graph_search, depth_first_graph_search,
    depth_limited_search, iterative_deepening_search,
    astar_search, recursive_best_first_search, compare_searchers, InstrumentedProblem
)

from pathfinder_synth.synth_graphs import graph_factory, get_components


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
            return cost_so_far + (self.graph.get(A, target) or np.inf)
        else:
            return cost_so_far + 1  # +1 for signal processing

    def h(self, node):
        # The idea here is to choose the path with the least latency
        min_edge_cost = self.find_min_edge()
        remaining_steps_estimate = 1  # Fixed estimate for remaining steps, we could probably implement something a bit more involved

        return min_edge_cost * remaining_steps_estimate


def compare_graph_searchers(searchers, graph_types):
    problems = []
    headers = ['Searcher'] + graph_types

    # Create problems for each graph
    for graph_type in graph_types:
        graph = graph_factory(graph_type)
        components = get_components()
        problem = SynthGraphProblem('input', 'output', graph, components)
        problems.append(problem)

    # Compare searchers across all problems
    compare_searchers(problems, headers, searchers)


# Define named search functions
def depth_limited_search_custom(p):
    return depth_limited_search(p, limit=15)


def astar_search_custom(p):
    return astar_search(p, h=p.h)


def rbfs_search_custom(p):
    return recursive_best_first_search(p, h=p.h)


# Search algorithms to compare
searchers = [
    breadth_first_graph_search,
    depth_first_graph_search,
    depth_limited_search_custom,
    iterative_deepening_search,
    astar_search_custom,
    rbfs_search_custom
]

# Graph types to compare
graph_types = ['ideal_path', 'wrong_path', 'dead_end']

compare_graph_searchers(searchers, graph_types)
