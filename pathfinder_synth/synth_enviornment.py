from aima.agents import Environment


class SynthEnvironment(Environment):
    def __init__(self, graph):
        super().__init__()
        self.components = None
        self.graph = graph

    def percept(self, agent):
        """Return agent's location and neighbor nodes."""
        neighbors = self.graph.get(agent.location)
        neighbor_nodes = list(neighbors.keys()) if neighbors else []
        return agent.location, neighbor_nodes

    def execute_action(self, agent, action):
        """Move agent to the next valid component. Kill the agent if there are no moves."""
        if action is None:
            agent.alive = False
            agent.performance -= 10
        else:
            neighbors = self.graph.get(agent.location)
            if neighbors and action in neighbors:
                print(f"Routing signal from {agent.location} to {action}")
                agent.location = action
                agent.performance += 1  # Good move
            else:
                print(f"Invalid action: {action} from {agent.location}")
                agent.performance -= 1  # Bad move

    def is_done(self):
        return any(agent.location == 'output' for agent in self.agents)

    def default_location(self, thing):
        return 'input'