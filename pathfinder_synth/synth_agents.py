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

from aima.agents import Agent

class SynthUtilityBasedAgent(Agent):
    def __init__(self, graph, components):
        super().__init__(self.program)
        self.graph = graph
        self.components = components
        self.path = []
        self.visited = set()
        self.current_path_length = 0
        self.__name__ = "Utility-Based Agent"

    def program(self, percept):
        location, neighbors = percept

        if location == 'output':
            print("Goal state.")
            self.alive = False
            return None

        # get neighbors that are 'on' and not visited
        available_neighbors = [n for n in neighbors if self.components[n].status == 1 and n not in self.visited]

        if available_neighbors:
            # Evaluate each available neighbor based on the number of available neighbors
            neighbor_scores = {}
            for neighbor in available_neighbors:
                neighbor_neighbors = self.graph.get(neighbor, {})
                # Count how many 'on' and unvisited neighbors the neighbor has
                count = sum(1 for nn in neighbor_neighbors if self.components[nn].status == 1 and nn not in self.visited)
                neighbor_scores[neighbor] = count

            # Select the neighbor with the highest score (most potential extensions)
            best_neighbor = max(neighbor_scores, key=neighbor_scores.get)
            best_score = neighbor_scores[best_neighbor]

            print(f"At {location}, evaluating neighbors: {neighbor_scores}")
            print(f"Selected action: {best_neighbor} with score: {best_score}")

            # Update path and visited nodes
            self.path.append(best_neighbor)
            self.visited.add(best_neighbor)
            self.current_path_length += 1

            return best_neighbor
        else:
            # If no unvisited neighbors, check if 'output' is a neighbor
            if 'output' in neighbors and self.components['output'].status == 1:
                print(f"No unvisited neighbors from {location}. Moving to 'output'.")
                self.path.append('output')
                self.current_path_length += 1
                return 'output'
            else:
                # No available moves
                print(f"No available moves from {location}. Agent is stuck.")
                self.alive = False
                return None

def get_synth_utility_based_agent(graph):
    return SynthUtilityBasedAgent(graph, components)