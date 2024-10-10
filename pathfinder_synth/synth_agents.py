from random import choice
from enum import Enum
from aima.agents import Agent
from pathfinder_synth.synth_graphs import get_components

class Action(Enum):
    PROCESS = 'process'
    MOVE = 'move'
    STUCK = 'stuck'

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
        returns the last visited component or none

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



# Factories
def agent_factory(agent_type):
    def create_agent():
        if agent_type == 'Reflex':
            return SynthReflexAgent(agent_type)
        elif agent_type == 'Model':
            return SynthModelBasedAgent(agent_type)
        elif agent_type == 'Utility':
            return SynthUtilityBasedAgent(agent_type, get_components())

    return create_agent