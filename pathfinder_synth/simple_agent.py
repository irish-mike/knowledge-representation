from aima.agents import Agent


def simple_agent_program(percept):
    """Agent moves to the first available neighbor"""
    location, neighbors = percept
    return neighbors[0] or None # Return the first available neighbor or none

agent = Agent(program=simple_agent_program)