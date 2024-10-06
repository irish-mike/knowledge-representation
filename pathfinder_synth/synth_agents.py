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

    return Agent(program)
