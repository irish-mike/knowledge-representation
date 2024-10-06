from aima.agents import Agent
from pathfinder_synth.synth_graphs import get_components


class SynthReflexAgent(Agent):
    """Reflex Agent that moves to the first available neighbor that is 'on'."""

    def __init__(self, name, components):
        super().__init__(self.program)
        self.__name__ = name
        self.components = components

    def neighbor_on(self, neighbor):
        return self.components[neighbor].status

    def program(self, percept):
        location, neighbors = percept
        return next((n for n in neighbors if self.neighbor_on(n)), None)


class SynthModelBasedAgent(SynthReflexAgent):
    """Model-Based Agent with Backtracking."""

    def __init__(self, name, components):
        super().__init__(name, components)
        self.visited = set()
        self.path_stack = []

    def is_valid_neighbor(self, neighbor):
        return self.neighbor_on(neighbor) and neighbor not in self.visited

    def get_valid_moves(self, neighbors):
        return [neighbor for neighbor in neighbors if self.is_valid_neighbor(neighbor)]

    def update_model(self, location):
        if location not in self.visited:
            self.visited.add(location)
            self.path_stack.append(location)

    def update_and_get_valid_moves(self, percept):
        location, neighbors = percept
        self.update_model(location)  # Update internal model
        return self.get_valid_moves(neighbors)

    def go_back(self):
        if self.path_stack:
            self.path_stack.pop()
            if self.path_stack:
                return self.path_stack[-1]
        return None

    def program(self, percept):
        moves = self.update_and_get_valid_moves(percept)
        if moves:
            return moves[0] # return the first valid move
        else:
            return self.go_back() # dead end retrace steps


class SynthUtilityBasedAgent(SynthModelBasedAgent):
    """Utility-Based Agent with preference for certain nodes."""

    def __init__(self, name, components):
        super().__init__(name, components)

    def get_best_neighbor(self, neighbors):
        return max(neighbors, key=lambda n: self.components[n].utility)


    def program(self, percept):
        moves = self.update_and_get_valid_moves(percept)

        if moves:
            return self.get_best_neighbor(moves)
        else:
            return self.go_back()

# Factories
def agent_factory(agent_type):
    if agent_type == 'Reflex':
        return SynthReflexAgent(agent_type, get_components())
    if agent_type == 'Model':
        return SynthModelBasedAgent(agent_type, get_components())
    if agent_type == 'Utility':
        return SynthUtilityBasedAgent(agent_type, get_components())

