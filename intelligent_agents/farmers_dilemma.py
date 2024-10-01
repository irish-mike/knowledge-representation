# The percepts for the farmer agent are as follows:
# farmer_location, fox_location, chicken_location, grain_location

# The actions are move(animal=none), if an animal is specified it will be moved to the opposite side along with the farmer,
# If no animal is specified the farmer will move alone.


# The actions and precept sequence needed to solve the problem are as follows
# [Farmer=A, Fox=A, Chicken=A, Grain=A] -> move(chicken)
# [Farmer=B, Fox=A, Chicken=B, Grain=A] -> move()
# [Farmer=A, Fox=A, Chicken=B, Grain=A] -> move(fox)
# [Farmer=B, Fox=B, Chicken=B, Grain=A] -> move(chicken)
# [Farmer=A, Fox=B, Chicken=A, Grain=A] -> move(grain)
# [Farmer=B, Fox=B, Chicken=A, Grain=B] -> move()
# [Farmer=A, Fox=B, Chicken=A, Grain=B] -> move(chicken)
# [Farmer=B, Fox=B, Chicken=B, Grain=B] -> goal state


from aima.agents import Environment, TableDrivenAgentProgram, Agent, TraceAgent

def generate_percept_table(states_actions):
    percept_table = {}
    percept_history = []

    for state_action in states_actions:
        state, action = state_action

        percept_history.append((
            state["farmer"],
            state["fox"],
            state["chicken"],
            state["grain"],
        ))

        percept_table[tuple(percept_history.copy())] = action

    return percept_table


class FarmersDilemmaEnvironment(Environment):
        def __init__(self):
            super().__init__()
            self.state = {"farmer": "A", "fox": "A", "chicken": "A", "grain": "A"}

        def is_done(self):
            return all(value == "B" for value in self.state.values())

        def percept(self, agent):
            return (
                self.state['farmer'],
                self.state['fox'],
                self.state['chicken'],
                self.state['grain']
            )

        def update_location(self, agent):
            if self.state[agent] == "A":
                self.state[agent] = "B"
            elif self.state[agent] == "B":
                self.state[agent] = "A"

        def move(self, agent=None):
            self.update_location("farmer")
            if agent in self.state:
                self.update_location(agent)

        def execute_action(self, agent, action):
            self.move(action)
            print(f"Moving {action if action else 'farmer alone'}, the new state is {self.state}")

def main():
    states_actions = [
        ({"farmer": "A", "fox": "A", "chicken": "A", "grain": "A"}, "chicken"),
        ({"farmer": "B", "fox": "A", "chicken": "B", "grain": "A"}, ""),
        ({"farmer": "A", "fox": "A", "chicken": "B", "grain": "A"}, "fox"),
        ({"farmer": "B", "fox": "B", "chicken": "B", "grain": "A"}, "chicken"),
        ({"farmer": "A", "fox": "B", "chicken": "A", "grain": "A"}, "grain"),
        ({"farmer": "B", "fox": "B", "chicken": "A", "grain": "B"}, ""),
        ({"farmer": "A", "fox": "B", "chicken": "A", "grain": "B"}, "chicken"),
        ({"farmer": "B", "fox": "B", "chicken": "B", "grain": "B"}, "complete"),  # Goal state
    ]

    # Generate the percept_table based on state actions
    percept_table = generate_percept_table(states_actions)

    # Create the environment
    env = FarmersDilemmaEnvironment()

    # Create the program
    program = TableDrivenAgentProgram(percept_table)

    # Create the agent
    farmer_agent = Agent(program)
    traced_farmer_agent = TraceAgent(farmer_agent)

    # add agent to environment
    env.add_thing(traced_farmer_agent)

    # Run agent in environment
    while not env.is_done():
        percept = env.percept(traced_farmer_agent)
        action = traced_farmer_agent.program(percept)
        env.execute_action(traced_farmer_agent, action)

    print("Done!")

if __name__ == "__main__":
    main()

