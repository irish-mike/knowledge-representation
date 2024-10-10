import random

from aima.agents import Agent
from pathfinder_synth.synth_graphs import get_components

'''
This simple reflex agent has a hard time navigating in this environment, 
since it only knows about its current state, it cannot make informed decisions about where to move next.
It tries to move and since it does not specify where to move the environment randomly chooses for it.
'''
class SynthReflexAgent(Agent):

    def __init__(self, name):
        super().__init__(self.program)
        self.__name__ = name

    def program(self, percept):
        location, signal, possible_moves = percept

        if signal:
            return 'process', location

        return 'move', random.choice(possible_moves)


'''
This is a simple Model based agent, it can navigate better than the reflex agent 
since it knows where it has already been, it can avoid getting re visiting the same nodes and won't get stuck going in loops.
'''
class SynthModelBasedAgent(SynthReflexAgent):
    def __init__(self, name):
        super().__init__(name)
        self.previous_components = ['input']  # Initialize with starting component

    def get_unvisited(self, possible_moves):
        # Returns the difference between the possible moves and previous nodes.
        return [move for move in possible_moves if move not in self.previous_components]

    def backtrack(self, location):
        # returns the last visited component or none
        current_index = self.previous_components.index(location)
        return self.previous_components[current_index - 1] if current_index > 0 else None

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

        next_location = random.choice(unvisited_moves) if unvisited_moves else self.backtrack(location)

        self.previous_components.append(next_location)

        return ('move', next_location) if next_location else ('Stuck', location)


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
        Returns a list of active components from the given moves.
        """
        return [comp for comp in moves if self.components[comp].status]

    def get_highest_utility(self, moves):
        """
        Returns the move with the highest signal value.
        """
        return max(moves, key=lambda comp_name: self.components[comp_name].signal, default=None)

    def program(self, percept):
        location, signal, possible_moves = percept

        if signal:
            return 'process', location

        # Get the components that have not been visited
        unvisited = self.get_unvisited(possible_moves)

        # Exclude components that are inactive
        filtered = self.filter_active_components(unvisited)

        best = self.get_highest_utility(filtered)

        if best is None:
            best = self.backtrack(location)
            if best is None:
                return 'Stuck', location

        self.previous_components.append(best)

        return 'move', best



# Factories
def agent_factory(agent_type):
    if agent_type == 'Reflex':
        return SynthReflexAgent(agent_type)
    if agent_type == 'Model':
        return SynthModelBasedAgent(agent_type)
    if agent_type == 'Utility':
        return SynthUtilityBasedAgent(agent_type, get_components())

