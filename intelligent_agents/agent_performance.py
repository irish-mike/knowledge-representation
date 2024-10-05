""" B. Agent performance """
import random

from matplotlib import pyplot as plt

from aima.agents import compare_agents, RandomVacuumAgent, TableDrivenVacuumAgent, ReflexVacuumAgent, \
    ModelBasedVacuumAgent, TrivialVacuumEnvironment, VacuumEnvironment, Agent

# PEAS For each Agent

peasTable = {
    "RandomVacuumAgent": {"Performance": "Tile Cleaned +10, Movement Penalty -1", "Environment": "Trivial Vacuum Environment","Actuators": "Move, Clean","Sensors": "Location, Tile State"},
    "TableDrivenVacuumAgent": {"Performance": "Tile Cleaned +10, Movement Penalty -1", "Environment": "Trivial Vacuum Environment","Actuators": "Move, Clean","Sensors": "Location, Tile State"},
    "ReflexVacuumAgent": {"Performance": "Tile Cleaned +10, Movement Penalty -1", "Environment": "Trivial Vacuum Environment","Actuators": "Move, Clean","Sensors": "Location, Tile State"},
    "ModelBasedVacuumAgent": {"Performance": "Tile Cleaned +10, Movement Penalty -1", "Environment": "Trivial Vacuum Environment","Actuators": "Move, Clean","Sensors": "Location, Tile State, Internal State"}}

print(f"+{'-'*25}+{'-'*40}+{'-'*30}+{'-'*20}+{'-'*40}+")
print(f"|{'Agent':<25}|{'Performance':<40}|{'Environment':<30}|{'Actuators':<20}|{'Sensors':<40}|")
print(f"+{'-'*25}+{'-'*40}+{'-'*30}+{'-'*20}+{'-'*40}+")
for agent, details in peasTable.items():
    print(f"|{agent:<25}|{details['Performance']:<40}|{details['Environment']:<30}|{details['Actuators']:<20}|{details['Sensors']:<40}|")
print(f"+{'-'*25}+{'-'*40}+{'-'*30}+{'-'*20}+{'-'*40}+")


# Comparative Analysis
# The random vacuum agent is simple but inefficient.
# It may clean the same square multiple times or miss cleaning parts of the environment entirely.
# However, in this trivial environment, the simplicity can be effective, as it is likely to eventually clean all tiles.
#
# The table-driven vacuum agent will perform well in the trivial environment since it can be programmed to respond to every known situation.
# Like the random agent, it is also easy to implement, but it does not scale well.
# In a larger environment, the size of the table required becomes excessively large, making it unmanageable.
# It will also struggle in a dynamic environments, as it cannot handle unknown changes that weren't programmed into the table.
#
# The reflex vacuum agent is also effective in the trivial environment.
# This agent maps its action to the current percepts, e.g. "if dirty, clean; else, move."
# It is easier to implement than the table-driven agent, but less effective as it has no knowledge of the percept history.
# Meaning, it is likely to clean the same tile repeatedly or move inefficiently, potentially getting stuck looping between tiles.
# Similar to the table-driven agent, the reflex agent has more difficulty in a dynamic environment, since the returned action must be pre-defined based on its percepts.
#
# The model-based vacuum agent is the most complex but also the most efficient.
# By maintaining an internal model of the environment, it can make informed decisions about what actions to take.
# This allows it to learn in dynamic environments, unlike the other agents.
# It will also scale much better, as its decision-making is based on its model, rather than predefined actions.
# However, in such a trivial environment the complexity may be overkill.

# Below code taken from Ruairí D. O’Reilly's Solution
class OneDimensionalVacuumEnvironment(VacuumEnvironment):
    """A one-dimensional vacuum environment with n tiles."""

    def __init__(self, n_tiles=5):
        super().__init__()
        self.n_tiles = n_tiles
        self.status = {i: random.choice(['Clean', 'Dirty']) for i in range(n_tiles)}

    def percept(self, agent):
        """Return the agent's current location and the status of the tile."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Execute the action of the agent: Move or Clean."""
        if action == 'Left' and agent.location > 0:
            agent.location -= 1
            agent.performance -= 1  # Moving costs
        elif action == 'Right' and agent.location < self.n_tiles - 1:
            agent.location += 1
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                self.status[agent.location] = 'Clean'
                agent.performance += 10  # Cleaning reward

    def is_done(self):
        """The environment is done if all tiles are clean."""
        return all(state == 'Clean' for state in self.status.values())

    def default_location(self, agent):
        """Ensure the agent starts within valid tile range."""
        return random.choice(range(self.n_tiles))

# Factories
def random_vacuum_agent_factory():
    agent = RandomVacuumAgent()
    agent.__name__ = "RandomVacuumAgent"  # Add a custom name to identify this agent
    print(f"Created agent: {agent.__name__}")
    return agent

def table_driven_vacuum_agent_factory():
    agent = TableDrivenVacuumAgent()
    agent.__name__ = "TableDrivenVacuumAgent"  # Add a custom name to identify this agent
    print(f"Created agent: {agent.__name__}")
    return agent

def reflex_vacuum_agent_factory():
    agent = ReflexVacuumAgent()
    agent.__name__ = "ReflexVacuumAgent"  # Add a custom name to identify this agent
    print(f"Created agent: {agent.__name__}")
    return agent

def model_based_vacuum_agent_factory():
    agent = ModelBasedVacuumAgent()
    agent.__name__ = "ModelBasedVacuumAgent"  # Add a custom name to identify this agent
    print(f"Created agent: {agent.__name__}")
    return agent


# Define the environment factory
def env_factory_trivial_vac():
    return TrivialVacuumEnvironment()

# Environment factory for the one-dimensional vacuum environment
def env_factory_1d(n_tiles=5):
    return OneDimensionalVacuumEnvironment(n_tiles=n_tiles)


agent_factories = [
    random_vacuum_agent_factory,
    table_driven_vacuum_agent_factory,
    reflex_vacuum_agent_factory,
    model_based_vacuum_agent_factory
]

def run_agent_comparison():
  # Run the comparison between the agents
  results = compare_agents(env_factory_trivial_vac, agent_factories , n=10, steps=1000)

  # Loop through the results and print each agent's name and average score
  for agent, avg_score in results:
    print(f"Agent: {agent.__name__}, Average Score: {avg_score}")


run_agent_comparison()


def run_agent_comparison_visualise_results():
    # Line styles for different agents
    line_styles = ['-.', '--', ':', '-']

    # Step sizes (powers of 2: 2^1 to 2^8)
    step_sizes = [2 ** i for i in range(1, 9)]

    # Store results for each agent
    performance_results = {agent.__name__: [] for agent in agent_factories}

    # Run comparison for each step size
    for steps in step_sizes:
        results = compare_agents(TrivialVacuumEnvironment, agent_factories, n=10, steps=steps)
        for agent, avg_score in results:
            performance_results[agent.__name__].append(avg_score)
    # Plot results
    plt.figure(figsize=(10, 6))

    for agent, style in zip(performance_results, line_styles):
        plt.plot(step_sizes, performance_results[agent], label=agent, linestyle=style)

    # Plot formatting
    plt.title('Agent Performance Across Different Step Sizes (Powers of 2)')
    plt.xlabel('Step Size')
    plt.ylabel('Average Performance')
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()

run_agent_comparison_visualise_results()

agent_factories_part_c = [random_vacuum_agent_factory, reflex_vacuum_agent_factory, model_based_vacuum_agent_factory]


# Define the environment factory for the one-dimensional environment
def env_factory_1d(n_tiles=5):
    return OneDimensionalVacuumEnvironment(n_tiles=n_tiles)


# Use compare_agents to test agents in the one-dimensional vacuum environment
def compare_agents_in_1D_env(n_tiles, steps):
    # Pass the factories, not instances
    results = compare_agents(lambda: env_factory_1d(n_tiles), agent_factories_part_c,  # Pass factories
                             n=10, steps=steps)
    return results


""" B. Goal-based Agent """

"""
Converting a Reflex Agent into a Goal-Based Agent:
- Define a Goal: The agent's goal is to clean the entire environment. It should know when it's done (when all tiles are clean).
- Percept History: The agent should keep track of the state of the environment (which tiles are clean/dirty) and where it has been.
- World Model: The agent needs a model of the environment (e.g., a 1-dimensional grid) to understand how its movements affect the world.
- Action Planning: Before deciding on an action, the agent should evaluate if that action will lead to progress towards the goal (cleaning the environment).


Goal: The goal is to clean all the dirty tiles in the environment.

Percept History and World Model: We'll keep track of the environment's state (clean or dirty) and the vacuum's position on the grid.

Planning Mechanism: The agent will decide its next action based on the goal of cleaning the entire room. It will try to visit every tile and clean if necessary

"""


# Goal-Based Vacuum Agent Program with Complete Tile Tracking
def GoalBasedVacuumAgentProgram(n_tiles):
    """A goal-based vacuum agent for a one-dimensional environment."""
    state = {i: 'Unknown' for i in range(n_tiles)}  # Initialize state for all tiles
    goal = 'Clean'  # Goal is to clean the entire environment

    def program(percept):
        location, status = percept  # Get the location and status (clean/dirty)
        state[location] = status  # Update the state of the current tile

        # Check if the environment is fully clean
        if all(tile_status == 'Clean' for tile_status in state.values()):
            print("Goal achieved: The entire environment is clean!")
            return 'NoOp'  # Stop acting

        # If the current tile is dirty, clean it, print the state, and return 'Suck'
        if status == 'Dirty':
            print(f"Current state: {state}")
            return 'Suck'

        # If the current tile is clean, move to the next unexplored or dirty tile, print the state, and return 'Left' or 'Right'
        if location > 0 and state[location - 1] != 'Clean':  # Check left
            print(f"Current state: {state}, Moving Left")
            return 'Left'
        elif location < len(state) - 1 and state[location + 1] != 'Clean':  # Check right
            print(f"Current state: {state}, Moving Left")
            return 'Right'
        else:
            # If both sides are clean or unknown, move randomly (since it's a 1D environment)
            print(f"Current state: {state}, Moving Randomly")
            return random.choice(['Left', 'Right'])

    return program


"""
World Model & Percept History: The agent maintains a state dictionary that tracks the status of each tile (whether it's clean or dirty). It updates the state with each percept it receives.

Goal: The agent's goal is to clean all the dirty tiles. Once all tiles are clean, the agent will stop working.

Action Planning: The agent evaluates whether the current tile is dirty. If it is, the agent cleans it. If the tile is already clean, the agent moves to the next tile that might be dirty, based on its percept history (state). The agent chooses to move left or right depending on which neighboring tile it thinks is still dirty.

Stopping Condition: Once all tiles are clean, the agent stops and does nothing (NoOp).

Improvements Over Reflex Agent:

- Memory: The agent now keeps track of where it has been and which tiles are clean or dirty.
- Goal: It has a defined goal: cleaning the entire environment. It knows when it has achieved this goal and stops acting.
- Simple Planning: The agent evaluates which action is more likely to help it achieve its goal (moving to dirty tiles).

"""


# Test the Goal-Based Agent in the 1D environment
def test_goal_based_agent():
    env = OneDimensionalVacuumEnvironment(n_tiles=5)  # Create a 5-tile environment
    agent = Agent(program=GoalBasedVacuumAgentProgram(env.n_tiles))  # Create the agent
    agent.location = env.default_location(agent)  # Place the agent in the environment

    # Run the environment until all tiles are clean
    while not env.is_done():
        percept = env.percept(agent)  # Get the agent's percept
        action = agent.program(percept)  # Get the action based on the goal-based program
        if action == 'NoOp':
            break  # Stop if NoOp is returned
        env.execute_action(agent, action)  # Execute the action in the environment

    print(f"Final performance: {agent.performance}")


# Run the test
test_goal_based_agent()
