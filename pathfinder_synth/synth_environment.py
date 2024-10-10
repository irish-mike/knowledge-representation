import random

from aima.agents import Environment
from pathfinder_synth.synth_graphs import graph_factory, get_components


class SynthEnvironment(Environment):
    def __init__(self, graph, components):
        super().__init__()
        self.components = components
        self.graph = graph

    def percept(self, agent):

        location = agent.location

        neighbors = self.graph.get(location)
        neighbor_nodes = list(neighbors.keys()) if neighbors else []

        signal = self.components[agent.location].signal

        return location, signal, neighbor_nodes

    def execute_action(self, agent, action):

        key, value = action

        if key == "move":
            self.handle_move(agent, value)
        elif key == "process":
            self.handle_process(agent, value)
        else:
            self.handle_invalid_action(agent, action)

        agent.performance -= 1  # any action the agent takes incurs a penalty

    def handle_move(self, agent, component):

        # If the component is off, the environment won't let the agent through
        if not self.components[component].status:
            agent.performance -= 5  # additional penalty for trying to make an invalid move
            return

        agent.location = component

    def handle_process(self, agent, value):
        if self.components[value].signal:
            self.components[value].signal = False
            agent.performance += 5

    def handle_invalid_action(self, agent, action):
        # Agent has become self-aware and is trying to do actions outside the defined scope, kill it.
        agent.alive = False
        agent.performance -= 100

    def get_random_neighbor(self, location):
        neighbors = self.graph.get(location)
        return random.choice(list(neighbors.keys()))

    def get_random_parent(self, component):
        parents = []
        for node, neighbors in self.graph.items():
            if component in neighbors:
                parents.append(node)

        return random.choice(parents) if parents else None

    def is_done(self):
        for agent in self.agents:
            self.handle_goal_state(agent)

        return super().is_done()

    def handle_goal_state(self, agent):
        # agent has reached the output and completed the goal
        if agent.location == "output":
            agent.performance += 10
            agent.alive = False

    def default_location(self, thing):
        return "input"

def synth_environment_factory(graph_type):
    graph = graph_factory(graph_type)
    components = get_components()
    return SynthEnvironment(graph, components)
