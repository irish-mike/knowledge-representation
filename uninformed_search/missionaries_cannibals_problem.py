from aima.search import Problem, breadth_first_tree_search

class MissionariesAndCannibals(Problem):
    """
    Missionaries and Cannibals Problem:
    Three missionaries and three cannibals are on one side of a river,
    along with a boat that can hold one or two people.
    Find a way to get everyone to the other side without ever leaving a group of missionaries in one
    place outnumbered by the cannibals in that place.

    State Representation:
    Each state is represented as a tuple (missionaries_left, cannibals_left, boat_position)
    - missionaries_left: Number of missionaries on the left bank
    - cannibals_left: Number of cannibals on the left bank
    - boat_position: Position of the boat (1 for left bank, 0 for right bank)
    """

    def __init__(self, initial=(3, 3, 1), goal=(0, 0, 0)):
        super().__init__(initial, goal)
        self.possible_moves = self._define_moves()

    def actions(self, state):
        direction = self._get_direction(state[2])  # Get direction based on boat position
        moves =  [
            move for move in self.possible_moves
            if self._is_action_valid(state, move, direction)
        ]
        return moves

    def result(self, state, action):
        direction = self._get_direction(state[2])  # Get direction based on boat position
        result_of_action = self._apply_action(state, action, direction)
        return result_of_action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    # Helper functions to adhere to SOLID principles

    def _define_moves(self):
        """Define the valid moves: 2 missionaries, 2 cannibals, or a mix."""
        return [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]

    def _get_direction(self, boat_position):
        """Determine the direction of movement based on boat position."""
        return -1 if boat_position == 1 else 1  # 1 means moving from left to right

    def _apply_action(self, state, action, direction):
        """Apply the action to return the new state."""
        missionaries_left, cannibals_left, boat_position = state
        missionaries_to_move, cannibals_to_move = action

        # Update number of missionaries and cannibals on the left bank
        new_missionaries_left = missionaries_left + direction * missionaries_to_move
        new_cannibals_left = cannibals_left + direction * cannibals_to_move

        # Toggle the boat position
        new_boat_position = 1 if boat_position == 0 else 0

        return (new_missionaries_left, new_cannibals_left, new_boat_position)

    def _is_action_valid(self, state, move, direction):
        """Check if an action is valid given the current state and direction."""
        new_state = self._apply_action(state, move, direction)
        return self._is_valid_state(new_state)

    def _is_valid_state(self, state):
        """Check if a state is valid (missionaries aren't outnumbered)."""
        missionaries_left, cannibals_left, _ = state
        missionaries_right = 3 - missionaries_left
        cannibals_right = 3 - cannibals_left

        if not (0 <= missionaries_left <= 3 and 0 <= cannibals_left <= 3):
            return False
        if not (0 <= missionaries_right <= 3 and 0 <= cannibals_right <= 3):
            return False
        if (missionaries_left > 0 and missionaries_left < cannibals_left) or \
           (missionaries_right > 0 and missionaries_right < cannibals_right):
            return False
        return True

problem = MissionariesAndCannibals()
solution_node = breadth_first_tree_search(problem)

path = solution_node.path()
print(f"Start at the initial state")

for i, node in enumerate(path):
    state = node.state
    if node.action:
        action = node.action
        print(f"Step {i}: Move {action[0]} missionaries and {action[1]} cannibals")

    print(
        f"State: Missionaries Left: {state[0]}, Cannibals Left: {state[1]}, Boat Position: {'Left' if state[2] == 1 else 'Right'}\n")
