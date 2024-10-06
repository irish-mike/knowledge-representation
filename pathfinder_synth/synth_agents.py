from aima.agents import Agent
from pathfinder_synth.synth_graphs import components


def get_synth_reflex_agent():
    def program(percept):
        """Agent moves to the first available neighbor that is 'on'."""
        location, neighbors = percept
        for neighbor in neighbors:
            if components[neighbor].status == 1:  # Check if the neighbor is 'on'
                return neighbor
        print(f"No available moves from {location}")
        return None

    agent = Agent(program)
    agent.__name__ = "Reflex Agent"

    return agent

def get_synth_model_based_agent():
    # let's keep track of the visited nodes
    visited = set()

    # we also need to keep track of the path the agent has taken
    path_stack = []

    def is_valid_neighbor(neighbor):
        """A neighbor is valid if it is on and not visited"""
        return components[neighbor].status == 1 and neighbor not in visited

    def get_valid_moves(neighbors):
        """get all valid neighbors"""
        return [neighbor for neighbor in neighbors if is_valid_neighbor(neighbor)]

    def update_model(location):
        """Update the agents internal model"""
        if location not in visited:
            visited.add(location)
            path_stack.append(location)

    def go_back():
        if path_stack:
            path_stack.pop()
            return path_stack[-1] or None
        else:
            return None

    def program(percept):
        location, neighbors = percept
        update_model(location) # Update internal model
        valid_moves = get_valid_moves(neighbors)# Get valid moves

        if valid_moves:
            return valid_moves[0]
        else:
            print(f"Hit Dead end, moving back")
            return go_back()

    agent = Agent(program)
    agent.__name__ = "Model Based Agent with Backtracking"

    return agent
