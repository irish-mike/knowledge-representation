# import itertools
# from aima.logic4e import FolKB, expr, fol_fc_ask, conjuncts, is_symbol, subst, unify, variables, constant_symbols
# from aima.utils4e import Expr
#
#
# def is_definite_clause(s):
#     """Allow negated facts in addition to definite clauses."""
#     if s.op == '~':  # Allow negated facts
#         return isinstance(s.args[0], Expr)
#     elif s.op == '==>':  # Implication operator
#         antecedent, consequent = s.args
#         antecedent_valid = all(isinstance(arg, Expr) for arg in conjuncts(antecedent))
#         consequent_valid = isinstance(consequent, Expr)
#         return antecedent_valid and consequent_valid
#     elif is_symbol(s.op):
#         return True  # Positive facts
#     else:
#         return False
#
#
# def parse_definite_clause(s):
#     """Return the antecedents and the consequent of a definite clause."""
#     if is_symbol(s.op) or s.op == '~':
#         return [], s
#     elif s.op == '==>':
#         antecedent, consequent = s.args
#         return conjuncts(antecedent), consequent
#     else:
#         raise ValueError(f"Clause is not a definite clause: {s}")
#
#
# def fol_fc_ask(KB, alpha):
#     """A simple forward-chaining algorithm. [Figure 9.3]"""
#     kb_consts = list({c for clause in KB.clauses for c in constant_symbols(clause)})
#
#     def enum_subst(p):
#         query_vars = list({v for clause in p for v in variables(clause)})
#         for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
#             theta = {x: y for x, y in zip(query_vars, assignment_list)}
#             yield theta
#
#     for q in KB.clauses:
#         phi = unify(q, alpha, {})
#         if phi is not None:
#             yield phi
#
#     while True:
#         new = []
#         for rule in KB.clauses:
#             p, q = parse_definite_clause(rule)
#             for theta in enum_subst(p):
#                 if set(subst(theta, p)).issubset(set(KB.clauses)):
#                     q_ = subst(theta, q)
#                     if all([unify(x, q_, {}) is None for x in KB.clauses + new]):
#                         new.append(q_)
#                         phi = unify(q_, alpha, {})
#                         if phi is not None:
#                             yield phi
#         if not new:
#             break
#         for clause in new:
#             KB.tell(clause)
#     return None
#
#
# class ExtendedFolKB(FolKB):
#     def tell(self, sentence):
#         """Override to handle negated facts."""
#         if is_definite_clause(sentence):
#             self.clauses.append(sentence)
#         else:
#             raise Exception(f"Not a definite clause: {sentence}")
#
#
# def Neq(x, y):
#     """Helper predicate for inequality."""
#     return expr(f'{x} != {y}')
#
#
# def query_and_infer(kb, query):
#     """Query the knowledge base and display results."""
#     answers = list(fol_fc_ask(kb, expr(query)))
#     result = f"Query: {query}, Result: "
#     if answers:
#         if answers == [{}]:  # If the answer list contains an empty dictionary
#             result += "True"
#         else:
#             result += ", ".join(str(answer) for answer in answers)
#     else:
#         result += "False"
#     print(result)
#
#
# """
#      Alice - Bob
#            |
#         ---------
#         |       |
#       Carol    Dave - Eve
#         |       |
#       Frank   George // Adding to test cousins
# """
#
#
# facts = [
#     # Carol's parents are Alice and Bob
#     'Parent(Alice, Carol)',
#     'Parent(Bob, Carol)',
#
#     # Dave's parents are also Alice and Bob
#     'Parent(Alice, Dave)',
#     'Parent(Bob, Dave)',
#
#     # Carol is the parent of Frank
#     'Parent(Carol, Frank)',
#
#     # Dave is the parent of George
#     'Parent(Dave, George)',
#
#     # Eve is Dave's spouse
#     'Spouse(Eve, Dave)',
#
#     # Carol and Dave are not the same person
#     'Neq(Carol, Dave)',
# ]
#
#
# rules = [
#     # Ancestor rules
#     'Parent(x, y) ==> Ancestor(x, y)',
#     '(Parent(x, z) & Ancestor(z, y)) ==> Ancestor(x, y)',
#
#     # Siblings rule
#     '(Parent(p, x) & Parent(p, y) & Neq(x, y)) ==> Siblings(x, y)',
#
#     '(Parent(px, x) & Parent(py, y) & Siblings(px, py)) ==> Cousin(x, y)',
# ]
#
#
#
#
#
#
# # Initialize KB
from aima.logic4e import FolKB, fol_fc_ask
from aima.utils4e import expr

# Initialize knowledge base
kb = FolKB()

# Define equality and inequality rules
kb.tell(expr("Eq(x, y)"))  # Equal
kb.tell(expr("Neq(x, z)"))  # Not Equal
kb.tell(expr("(Eq(x, y) & Neq(y, z)) ==> XOR(x, z)"))  # Exclusive OR rule

# Add family facts
kb.tell(expr("Parent(Alice, Carol)"))
kb.tell(expr("Parent(Alice, Dave)"))

kb.tell(expr("Parent(Bob, Carol)"))
kb.tell(expr("Parent(Bob, Dave)"))

# Carol and Dave are not the same person
kb.tell(expr("Neq(Carol, Dave)"))

# Same mother rule
kb.tell(expr("(Parent(p1, x) & Parent(p1, y) & Neq(x, y)) ==> SameMother(x, y)"))

# XOR Test
print("XOR Test")
answers = list(fol_fc_ask(kb, expr("XOR(Carol, Dave)")))
print("Query: XOR(Carol, Dave)")
print("Result:", "True" if answers else "False")

# SameMother Test
print("\nSameMother Test")
answers = list(fol_fc_ask(kb, expr("SameMother(Carol, Dave)")))
print("Query: SameMother(Carol, Dave)")
print("Result:", "True" if answers else "False")
