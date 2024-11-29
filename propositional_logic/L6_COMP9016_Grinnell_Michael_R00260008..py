from aima.logic import Expr, tt_entails, PropKB

def evaluate(statement, conclusion, message):
    result = tt_entails(statement, conclusion)
    print(f"{message}: {result}")


def check_rain_scenario():
    """
    Checks if the premises:
    1. If it rains, then the ground is wet (R => W)
    2. It is raining (R)
    entail the conclusion:
    - The ground is wet (W).
    """
    R = Expr('R')  # It is raining
    W = Expr('W')  # The ground is wet

    # Define the premises
    premise1 = Expr('==>', R, W)  # If it rains, then the ground is wet
    premise2 = R  # It is raining

    statement = premise1 & premise2
    conclusion = W

    evaluate(statement, conclusion, "Do the premises entail the conclusion?")

def check_logical_entailment_examples():
    """
    Checks various entailment examples to verify logical relationships:
    - A ∧ B |= A
    - A ∨ B |= B ∨ A
    - A ⇒ B |= ¬A ∨ B
    """
    A, B = map(Expr, 'AB')

    # Example 1: A & B |= A
    evaluate((A & B), A, "A & B entails A")

    # Example 2: A | B |= B | A
    evaluate((A | B), (B | A), "A | B entails B | A")

    # Example 3: A ⇒ B |= ¬A ∨ B

    statement = Expr('==>', A, B)
    conclusion = ~A | B
    evaluate(statement, conclusion, "A ==> B entails ¬A | B")


def check_study_msc_scenario():
    """
    Checks if the premises:
    1. If you study, you will pass the exam (S => PE)
    2. If you pass the exam, you will get an MSc in AI (PE => MSC)
    3. You study (S)
    entail the conclusion:
    - You will get an MSc in AI (MSC).
    """
    S = Expr('S')  # You study
    PE = Expr('PE')  # You pass the exam
    MSC = Expr('MSC')  # You get an MSc in AI

    # Define the premises
    premise1 = Expr('==>', S, PE)  # If you study, you will pass the exam
    premise2 = Expr('==>', PE, MSC)  # If you pass, you will get an MSc in AI
    premise3 = S  # You study

    # Combine the premises
    statement = premise1 & premise2 & premise3

    # Define the conclusion
    conclusion = MSC  # You will get an MSc in AI

    # Check if the premises entail the conclusion
    evaluate(statement, conclusion, "If you study, you will get an MSc in AI")

def setup_wumpus_world():
    """
    Sets up a 3x3 Wumpus World with a pit at [2,2] and breezes in adjacent squares.
    Prints ASCII representation of the world.
    """
    wumpus_kb = PropKB()

    # Define pit symbols
    P11, P12, P13, P21, P22, P23, P31, P32, P33 = map(Expr, [
        'P11', 'P12', 'P13',
        'P21', 'P22', 'P23',
        'P31', 'P32', 'P33'
    ])

    # Define breeze symbols
    B11, B12, B13, B21, B22, B23, B31, B32, B33 = map(Expr, [
        'B11', 'B12', 'B13',
        'B21', 'B22', 'B23',
        'B31', 'B32', 'B33'
    ])

    # Add clauses to the knowledge base
    wumpus_kb.tell(~P11)  # No pit in [1,1]
    wumpus_kb.tell(~B11)  # No breeze in [1,1]
    wumpus_kb.tell(P22)  # Pit in [2,2]

    # Breezes adjacent to pits using biconditionals
    wumpus_kb.tell(Expr('<=>', B11, P12 | P21))
    wumpus_kb.tell(Expr('<=>', B12, P11 | P22 | P13))
    wumpus_kb.tell(Expr('<=>', B13, P12 | P23))

    wumpus_kb.tell(Expr('<=>', B21, P11 | P22 | P31))
    wumpus_kb.tell(Expr('<=>', B22, P12 | P21 | P23 | P32))
    wumpus_kb.tell(Expr('<=>', B23, P13 | P22 | P33))

    wumpus_kb.tell(Expr('<=>', B31, P21 | P32))
    wumpus_kb.tell(Expr('<=>', B32, P22 | P31 | P33))
    wumpus_kb.tell(Expr('<=>', B33, P23 | P32))

    # Print clauses in knowledge base
    print("Clauses in the knowledge base:")
    for clause in wumpus_kb.clauses:
        print(clause)

    # Generate ASCII representation of the Wumpus World
    world = [[' ' for _ in range(3)] for _ in range(3)]
    world[0][0] = 'P' if wumpus_kb.ask_if_true(P11) else 'B' if wumpus_kb.ask_if_true(B11) else '.'
    world[0][1] = 'P' if wumpus_kb.ask_if_true(P12) else 'B' if wumpus_kb.ask_if_true(B12) else '.'
    world[0][2] = 'P' if wumpus_kb.ask_if_true(P13) else 'B' if wumpus_kb.ask_if_true(B13) else '.'
    world[1][0] = 'P' if wumpus_kb.ask_if_true(P21) else 'B' if wumpus_kb.ask_if_true(B21) else '.'
    world[1][1] = 'P' if wumpus_kb.ask_if_true(P22) else 'B' if wumpus_kb.ask_if_true(B22) else '.'
    world[1][2] = 'P' if wumpus_kb.ask_if_true(P23) else 'B' if wumpus_kb.ask_if_true(B23) else '.'
    world[2][0] = 'P' if wumpus_kb.ask_if_true(P31) else 'B' if wumpus_kb.ask_if_true(B31) else '.'
    world[2][1] = 'P' if wumpus_kb.ask_if_true(P32) else 'B' if wumpus_kb.ask_if_true(B32) else '.'
    world[2][2] = 'P' if wumpus_kb.ask_if_true(P33) else 'B' if wumpus_kb.ask_if_true(B33) else '.'

    # Print the ASCII representation of the world
    print("\nASCII representation of the world:")
    for row in world:
        print(' '.join(row))



check_rain_scenario()
check_logical_entailment_examples()
check_study_msc_scenario()
setup_wumpus_world()
