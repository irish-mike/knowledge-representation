import sys
import time

import numpy as np
from aima.search import (
    GraphProblem,
    breadth_first_graph_search, depth_first_graph_search,
    depth_limited_search, iterative_deepening_search,
    astar_search, recursive_best_first_search, compare_searchers, InstrumentedProblem, breadth_first_tree_search,
    greedy_best_first_graph_search
)
from aima.utils import print_table

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


def average_search_time(problem, search_function, runs):
    total_time = 0

    for _ in range(runs):
        # Start the timer
        start_time = time.time()

        # Perform the search
        result = search_function(problem)

        # Stop the timer
        end_time = time.time()

        # Calculate time for this run
        total_time += (end_time - start_time)

    # Calculate the average time
    average_time = total_time / runs

    return average_time

def compare_graph_searchers(searchers, graph_types, runs=100):
    problems = []
    headers = ['Searcher', 'Graph Type', 'Average Nodes Expanded', 'Solution Length', 'Path Cost', 'Average Time (s)']

    # Create problems for each graph
    for graph_type in graph_types:
        graph = graph_factory(graph_type)
        components = get_components(graph_type)
        problem = SynthGraphProblem('input', 'output', graph, components)
        problems.append((graph_type, problem))  # Store graph type and problem

    # Table to store results
    results = []

    # Iterate over each searcher and problem
    for searcher in searchers:
        searcher_name = searcher.__name__  # Get the searcher's name

        for graph_type, problem in problems:
            # InstrumentedProblem tracks metrics (e.g., nodes expanded)
            inst_problem = InstrumentedProblem(problem)

            # Get the average time for this searcher and problem type
            avg_time = average_search_time(inst_problem, searcher, runs=runs)

            # Run the search once to collect other metrics (nodes expanded, etc.)
            search_result = searcher(inst_problem)

            # Collect metrics: nodes expanded and solution length
            nodes_expanded = int(inst_problem.succs / runs)  # Average Number of nodes expanded
            solution_length = len(search_result.solution()) if search_result else 'N/A'  # Solution length

            # Calculate path cost if a solution exists
            if search_result:
                path_cost = search_result.path_cost  # Assuming Node.path_cost stores the total cost
            else:
                path_cost = 'N/A'

            # Append a row of results
            results.append([searcher_name, graph_type, nodes_expanded, solution_length, path_cost, f"{avg_time:.6f} seconds"])

    # Print the results table
    print_table(results, headers)


# Define named search functions
def depth_limited_search_factory(p):
    return depth_limited_search(p, limit=7)

def astar_search_factory(p):
    return astar_search(p, h=p.h)

def greedy_best_first_graph_search_factory(p):
    return greedy_best_first_graph_search(p, f=p.h)

def recursive_best_first_search_search_factory(p):
    return recursive_best_first_search(p, h=p.h)

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

# Compare the searchers and print the results
compare_graph_searchers(searchers, graph_types)

