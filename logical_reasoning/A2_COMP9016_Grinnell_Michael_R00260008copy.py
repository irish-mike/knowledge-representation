import itertools
import random

import aima

import random

from aima.logic import conjuncts
from aima.logic4e import is_symbol, FolKB, fol_fc_ask, fol_bc_ask
from aima.utils4e import Expr, expr

# overridden to include negated facts (~Fact)
def is_definite_clause(s):

    if is_symbol(s.op):
        return True

    if s.op == '~':
        return isinstance(s.args[0], Expr)

    if s.op == '==>':
        antecedent, consequent = s.args

        # The old definition would fail when checking if ~ is a symbol
        # so here we check if each part of teh sentence is valid using isinstance of Expr
        antecedent_valid = all(isinstance(arg, Expr) for arg in conjuncts(antecedent))
        consequent_valid = isinstance(consequent, Expr)
        return antecedent_valid and consequent_valid

    return False

# Overriding this function to handle negated facts (~Fact).
def parse_definite_clause(s):

    if is_symbol(s.op) or s.op == '~':
        return [], s

    if s.op == '==>':
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

# Monkey patch to override aima definitions
aima.logic4e.is_definite_clause = is_definite_clause
aima.logic4e.parse_definite_clause = parse_definite_clause


"""
Problem Statement: Define the following family relationships and property inheritance rules
in first-order logic. Use these definitions to create a knowledge base and infer new relationships
and properties.

Given the following facts:
• Alice and Bob are parents of Carol.
• Alice and Bob are parents of Dave.
• Eve is the spouse of Dave.
• Carol is the parent of Frank.
• Dave is the parent of George.
• Carol has blue eyes.
"""

"""
     Alice - Bob 
           |
        ---------
        |       |
      Carol    Dave - Eve
        |       |
      Frank   George // Adding to test cousins
"""

# Define constants (individuals in the domain)
constants = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'George']

# Define predicates (relationships and properties)
predicates = [
    'Parent(x, y)',     # x is the parent of y
    'Ancestor(x, y)',   # x is an ancestor of y
    'Cousin(x, y)',     # x is a cousin of y
    'Siblings(x, y)',   # x and y are siblings
    'Spouse(x, y)',     # x and y are spouses
    '~Same(x, y)'       # x and y are not the same person
    'BlueEyes(x)',      # x has blue eyes
]

# Define facts (base knowledge in the domain)
facts = [
    'Parent(Alice, Carol)',  # Alice is a parent of Carol
    'Parent(Bob, Carol)',    # Bob is a parent of Carol
    'Parent(Alice, Dave)',   # Alice is a parent of Dave
    'Parent(Bob, Dave)',     # Bob is a parent of Dave
    'Spouse(Eve, Dave)',     # Eve is the spouse of Dave
    'Parent(Carol, Frank)',  # Carol is a parent of Frank
    'Parent(Dave, George)',  # Dave is a parent of George
    'BlueEyes(Carol)',       # Carol has blue eyes
    '~BlueEyes(Dave)',       # Dave does not have blue eyes
]



"""
1. Ancestor Relationship:
   - A person is an ancestor of another if they are a parent or if they are a parent of one of 
     the other person’s ancestors.

2. Cousin Relationship:
   - Two people are cousins if their parents are siblings and they are not the same person.

3. Sibling Relationship:
   - Two people are siblings if they share the same parent and are not the same person.
"""

# Define rules (logic to infer new relationships)
rules = [
    # Ancestor rules
    'Parent(x, y) ==> Ancestor(x, y)',
    '(Parent(x, z) & Ancestor(z, y)) ==> Ancestor(x, y)',

    # Siblings rule
    '(Parent(p, x) & Parent(p, y) & ~Same(x, y)) ==> Siblings(x, y)',

    # Cousins rule
    '(Parent(px, x) & Parent(py, y) & Siblings(px, py)) ==> Cousin(x, y)',

    # Inheritance rule
    '(Parent(x, y) & BlueEyes(x)) ==> ChanceOfBlueEyes(y)',
    '(Parent(x, y) & BlueEyes(y)) ==> ChanceOfBlueEyes(x)',

    # Has an ancestor with blue eyes
    '(Ancestor(x, y) & BlueEyes(x)) ==> AncestorWithBlueEyes(y)',
]


# Function to add "not the same" facts to the KB
def tell_not_same(kb):
    for i in range(len(constants)):
        for j in range(len(constants)):
            if i != j:
                kb.tell(expr(f"~Same({constants[i]}, {constants[j]})"))




def assign_blue_eyes(kb):

    # Get all the people where at least one parent has blue eyes
    query = "ChanceOfBlueEyes(x)"
    answers = list(fol_fc_ask(kb, expr(query)))

    # With a 50% chance tell the kb that individual has blue eyes
    for answer in answers:
        person  = answer[expr('x')]
        if random.random() < 0.5:
            kb.tell(expr(f"BlueEyes({person})"))
        else:
            kb.tell(expr(f"~BlueEyes({person})"))


# Initialize KB
kb = FolKB()

# Add rules to KB
for rule in rules:
    kb.tell(expr(rule))

tell_not_same(kb)

# Add facts to KB
for fact in facts:
    kb.tell(expr(fact))

# Assign blue eyes based on "ChanceOfBlueEyes"
assign_blue_eyes(kb)

def query_and_infer(kb, query, technique):
    if technique == "forward_chaining":
        answers = list(fol_fc_ask(kb, expr(query)))
    elif technique == "backward_chaining":
        answers = list(fol_bc_ask(kb, expr(query)))

    result = "True" if answers else "False"
    return result, answers


def display_results(question, query, technique, expected, result):
    print(f"| {question:<35} | {query:<30} | {technique:<20} | {expected:<8} | {result:<8} |")


def display_answers(answers):
    answers = [answer for answer in answers if answer] # remove empty objects
    if answers:
        print("Results:")
        for idx, answer in enumerate(answers, 1):
            print(f"    {idx}. {answer}")


def query_and_show_table(kb, query, technique, question, expected):
    result, answers = query_and_infer(kb, query, technique)
    display_results(question, query, technique, expected, result)
    display_answers(answers)



def print_header():
    """Print the table header."""
    print("+-------------------------------------+--------------------------------+----------------------+----------+----------+")
    print("| Question                            | Query                          | Technique            | Expected | Result   |")
    print("+-------------------------------------+--------------------------------+----------------------+----------+----------+")


def print_footer():
    """Print the table footer."""
    print("+-------------------------------------+--------------------------------+----------------------+----------+----------+")


# Queries
print("\n--- Testing Knowledge Base Questions ---\n")
print_header()
# Parent Queries
query_and_show_table(kb, 'Parent(Alice, Carol)', "forward_chaining", "Is Alice the parent of Carol?", "True")
query_and_show_table(kb, 'Parent(Bob, Frank)', "backward_chaining", "Is Bob the parent of Frank?", "False")

# Ancestor Queries
query_and_show_table(kb, 'Ancestor(Alice, Carol)', "forward_chaining", "Is Alice an ancestor of Carol?", "True")
query_and_show_table(kb, 'Ancestor(Bob, Frank)', "backward_chaining", "Is Bob an ancestor of Frank?", "True")
query_and_show_table(kb, 'Ancestor(Bob, Eve)', "forward_chaining", "Is Bob an ancestor of Eve?", "False")

# Sibling Queries
query_and_show_table(kb, 'Siblings(Carol, Dave)', "forward_chaining", "Are Carol and Dave siblings?", "True")
query_and_show_table(kb, 'Siblings(Frank, George)', "backward_chaining", "Are Frank and George siblings?", "False")
query_and_show_table(kb, 'Siblings(Dave, Dave)', "forward_chaining", "Are Dave and Dave siblings?", "False")

# Cousin Queries
query_and_show_table(kb, 'Cousin(Frank, George)', "forward_chaining", "Are Frank and George cousins?", "True")
query_and_show_table(kb, 'Cousin(Carol, George)', "backward_chaining", "Are Carol and George cousins?", "False")
query_and_show_table(kb, 'Cousin(Dave, Dave)', "forward_chaining", "Are Dave and Dave cousins?", "False")

# Blue Eyes Queries
query_and_show_table(kb, 'BlueEyes(Carol)','forward_chaining', 'Does Carol have blue eyes?', 'True')
query_and_show_table(kb, 'BlueEyes(Dave)','forward_chaining', 'Does Dave have blue eyes?', 'False')

print_footer()


print("\n--- Specifically Asked Questions ---\n")
print_header()

query_and_show_table(kb, 'BlueEyes(Frank)','forward_chaining', 'Does Frank have blue eyes?', '?')
query_and_show_table(kb, 'BlueEyes(Frank)','backward_chaining', 'Does Frank have blue eyes?', '?')
query_and_show_table(kb, 'AncestorWithBlueEyes(Frank)','forward_chaining', 'Frank has ancestor with blue eyes?', 'True')
query_and_show_table(kb, 'Cousin(Carol, Eve)', "forward_chaining", "Are Carol and Eve cousins?", "False")

print_footer()

def derive_all_inferences(kb):
    print("\n--- Deriving All Possible Inferences ---")

    queries = [
        ('Parent(x, y)', "x is a parent of y"),
        ('Ancestor(x, y)', "x is an ancestor of y"),
        ('Siblings(x, y)', "x is a sibling of y"),
        ('Cousin(x, y)', "x and y are cousins"),
        ('BlueEyes(x)', "x has blue eyes"),
    ]

    for query, description in queries:
        print(f"\n--- {description} ---")
        results = list(fol_fc_ask(kb, expr(query)))
        if results:
            for idx, result in enumerate(results, 1):
                print(f"{idx}. {result}")
        else:
            print("No inferences found.")

# Example call to the function
derive_all_inferences(kb)



