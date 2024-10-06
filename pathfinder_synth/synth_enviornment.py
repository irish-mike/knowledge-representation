from aima.agents import Environment


class SynthEnvironment(Environment):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def percept(self, agent):
        """Return agent's location and neighbor nodes."""
        neighbors = self.graph.get(agent.location)
        if neighbors:
            neighbor_nodes = list(neighbors.keys())  # Neighbors are the dictionary keys
        else:
            neighbor_nodes = []
        return agent.location, neighbor_nodes

    def execute_action(self, agent, action):
        neighbors = self.graph.get(agent.location)
        if neighbors and action in neighbors:
            print(f"Routing signal from {agent.location} to {action}")
            agent.location = action
            agent.performance += 1  # Increment performance for a successful move
        else:
            print(f"Invalid action: {action} from {agent.location}")
            agent.performance -= 1  # Decrement performance for an invalid action

    def is_done(self):
        return any(agent.location == 'output' for agent in self.agents)

    def default_location(self, thing):
        return 'input'