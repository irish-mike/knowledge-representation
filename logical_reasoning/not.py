import itertools
from aima.logic4e import FolKB, expr, fol_fc_ask, conjuncts, is_symbol, subst, unify, variables, constant_symbols
from aima.utils4e import Expr

def is_definite_clause(s):
    """Allow negated facts in addition to definite clauses."""
    if s.op == '~':  # Allow negated facts
        return isinstance(s.args[0], Expr)
    elif s.op == '==>':  # Implication operator
        antecedent, consequent = s.args
        antecedent_valid = all(isinstance(arg, Expr) for arg in conjuncts(antecedent))
        consequent_valid = isinstance(consequent, Expr)
        return antecedent_valid and consequent_valid
    elif is_symbol(s.op):
        return True  # Positive facts
    else:
        return False

def parse_definite_clause(s):
    """Return the antecedents and the consequent of a definite clause."""
    # Handle facts and negated facts
    if is_symbol(s.op) or s.op == '~':
        return [], s
    elif s.op == '==>':
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent
    else:
        raise ValueError(f"Clause is not a definite clause: {s}")

def fol_fc_ask(KB, alpha):
    """A simple forward-chaining algorithm. [Figure 9.3]"""
    kb_consts = list({c for clause in KB.clauses for c in constant_symbols(clause)})

    def enum_subst(p):
        query_vars = list({v for clause in p for v in variables(clause)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    # check if we can answer without new inferences
    for q in KB.clauses:
        phi = unify(q, alpha, {})
        if phi is not None:
            yield phi

    while True:
        new = []
        for rule in KB.clauses:
            p, q = parse_definite_clause(rule)
            for theta in enum_subst(p):
                if set(subst(theta, p)).issubset(set(KB.clauses)):
                    q_ = subst(theta, q)
                    if all([unify(x, q_, {}) is None for x in KB.clauses + new]):
                        new.append(q_)
                        phi = unify(q_, alpha, {})
                        if phi is not None:
                            yield phi
        if not new:
            break
        for clause in new:
            KB.tell(clause)
    return None

# Initialize KB with modified clause handling
class ExtendedFolKB(FolKB):
    def tell(self, sentence):
        """Override to handle negated facts."""
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception(f"Not a definite clause: {sentence}")

# Populate KB with facts and rules
kb = ExtendedFolKB()
kb.tell(expr('Parent(Alice, Dave)'))           # Positive fact
kb.tell(expr('~Parent(Alice, Carol)'))        # Negated fact
kb.tell(expr('Parent(x, y) ==> Ancestor(x, y)'))  # Rule to infer ancestors
kb.tell(expr('(Parent(x, y) & Parent(y, z)) ==> Grandparent(x, z)'))
kb.tell(expr('~Ancestor(Alice, Eve)'))

# Define a function to query the KB with improved output
def query_kb(kb, query):
    """Query the knowledge base and display results."""
    answers = list(fol_fc_ask(kb, expr(query)))
    result = f"Query: {query}, Result: "
    if answers:
        if answers == [{}]:  # If the answer list contains an empty dictionary
            result += "True"
        else:
            result += ",".join(str(answer) for answer in answers)
    else:
        result += "False"
    print(result)




# Test NOT operator
print("\n--- Testing Knowledge Base ---")
query_kb(kb, '~Parent(Alice, Bob)')   # Should return true
query_kb(kb, 'Parent(Alice, Carol)') # Should return false (negated fact)
query_kb(kb, 'Ancestor(Alice, Bob)') # Should infer Alice is an ancestor of Bob
query_kb(kb, 'Parent(Alice, x)')  # Should return x = Bob
query_kb(kb, 'Ancestor(Alice, Eve)') # Should return false (negated fact)

# Add new rules and facts
kb.tell(expr('(Ancestor(x, y) & Parent(y, z)) ==> Ancestor(x, z)'))
kb.tell(expr('Parent(Bob, Dave)'))

# Test inference with new facts
query_kb(kb, 'Ancestor(Alice, Dave)') # Should infer Alice is an ancestor of Dave
