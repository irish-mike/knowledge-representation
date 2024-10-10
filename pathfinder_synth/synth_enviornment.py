import random

from aima.agents import Environment


class SynthEnvironment(Environment):
    def __init__(self, graph, components):
        super().__init__()
        self.components = components
        self.graph = graph

    def percept(self, agent):
        return agent.location, self.components[agent.location].signal

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

        # moves the agent randomly if location is nit specified
        if component not in self.components:
            component = self.get_random_neighbor(agent.location)

        # If the component is off, the environment won't let the agent through
        if not self.components[component].status:
            print(f"Component {component} is OFF, Cannot move agent")
            agent.performance -= (
                1  # additional penalty for trying to make an invalid move
            )
            return

        print(f"Routing signal from {agent.location} to {component}")

        agent.location = component

    def handle_process(self, agent, value):

        print(f"Processing signal at {value}")

        if self.components[value].signal:
            self.components[value].signal = False
            agent.performance += 5

    def handle_invalid_action(self, agent, action):
        # Agent has become self-aware and is trying to do actions outside the defined scope, kill it.
        print(f"Invalid action: {action} from {agent.location}")
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
            print(f"Agent has reached the output")
            agent.performance += 10
            agent.alive = False

    def default_location(self, thing):
        return "input"
