import numpy as np

from aima.search import GraphProblem, breadth_first_graph_search, depth_first_tree_search
from pathfinder_synth.synth_graphs import graph_factory, get_components


class SynthGraphProblem(GraphProblem):
    """A problem for searching through a synthesizer's signal path graph."""

    def __init__(self, initial, goal, graph, components):
        super().__init__(initial, goal, graph)
        self.components = components  # Dictionary of component states

    def actions(self, A):
        """Return available actions at the current component (node)."""
        actions = []

        # Check if there is a signal to process at the current component
        if self.components[A].signal:
            actions.append(('process', A))

        # Get neighboring components
        neighbors = super().actions(A)

        # Only allow movement to active neighbors
        for neighbor in neighbors:
            if self.components[neighbor].status:
                actions.append(('move', neighbor))

        return actions

    def result(self, state, action):
        """Return the next state after taking the given action."""
        action_type, target = action

        if action_type == 'move':
            return target  # New state is the target component

        return state  # If processing, state remains the same

    def path_cost(self, cost_so_far, A, action, B):
        """Return the cumulative cost after performing the action."""
        action_type, target = action
        if action_type == 'move':
            # Add the latency between components
            return cost_so_far + (self.graph.get(A, target) or np.inf)
        else:
            # Fixed cost for processing a signal
            return cost_so_far + 1

    def goal_test(self, state):
        """Check if the current state is the goal (reaching 'output')."""
        return state == self.goal

    def find_min_edge(self):
        """Find minimum value of edges for informed search algorithms."""
        return super().find_min_edge()

    def h(self, node):
        """Heuristic function for estimating cost to the goal."""
        # For simplicity, we'll use a heuristic of 0 or implement a custom heuristic
        return 0  # Replace with more meaningful heuristic if possible


initial_state = 'input'

goal_state = 'output'

# Create the graph
graph = graph_factory('ideal_path')

# Create an instance of the problem
problem = SynthGraphProblem(initial_state, goal_state, graph, get_components())

# Perform the search
solution_node = depth_first_tree_search(problem)

# Retrieve and display the solution
if solution_node:
    path = solution_node.path()
    print("Solution found:")
    for node in path:
        action = node.action
        state = node.state
        if action:
            action_type, target = action
            print(f"Action: {action_type}, Target: {target}, Location: {state}")
        else:
            # Initial state
            print(f"Start at {state}")
else:
    print("No solution found.")
