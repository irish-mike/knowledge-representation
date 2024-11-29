import aima
import random
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from aima.utils import Expr, expr
from itertools import combinations
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from aima.learning import NaiveBayesLearner, DataSet
from sklearn.model_selection import train_test_split
from aima.logic import conjuncts, is_symbol, FolKB, fol_fc_ask, fol_bc_ask
from aima.probability import BayesNet, enumeration_ask, gibbs_ask, likelihood_weighting


def print_section(title):
    print("\n" + "*" * 60)
    print(f"{title:^60}")
    print("*" * 60 + "\n")

print_section("ASSIGNMENT: KNOWLEDGE REPRESENTATION")

print(f"{'Name: Michael Grinnell':^60}")
print(f"{'Student Number: R00260008':^60}")
print("*" * 60 + "\n")


print_section("QUESTION 1: LOGICAL REASONING")

# I had some issues with the negation operator
# So I had to override some of the functions in the aima repo

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
aima.logic.is_definite_clause = is_definite_clause
aima.logic.parse_definite_clause = parse_definite_clause


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

print_section("QUESTION 1.4: PERFORMING INFERENCE")
print("\nUsing inference methods (forward and backward chaining) to derive:")
print("\n  - If Frank has blue eyes.")
print("\n  - If Frank has an ancestor with blue eyes.")
print("\n  - If Carol and Eve are cousins.")
print("\n  - All possible ancestor relationships.")

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

print_section("QUESTION 1.1.3: BUILDING KNOWLEDGE BASE")
print("\nDeriving all possible inferences from the given facts and rules..")
derive_all_inferences(kb)

# Question 2
print_section("QUESTION 2: BAYESIAN NETWORKS")

network = BayesNet([
    ('EconomyStrength', '', 0.6),  # EconomyStrength: True = Strong, False = Weak
    ('GovernmentPolicy', '', 0.7),  # GovernmentPolicy: True = Supportive, False = Unsupportive

    # JobMarket depends on EconomyStrength and TechInnovation
    ('TechInnovation', 'EconomyStrength GovernmentPolicy', {  # TechInnovation depends on EconomyStrength and GovernmentPolicy
        (True, True): 0.85,  # Strong Economy & Supportive Gov → High TechInnovation
        (True, False): 0.7,  # Strong Economy & Unsupportive Gov → High TechInnovation
        (False, True): 0.4,  # Weak Economy & Supportive Gov → High TechInnovation
        (False, False): 0.2  # Weak Economy & Unsupportive Gov → High TechInnovation
    }),

    ('JobMarket', 'EconomyStrength TechInnovation', {  # Job Market depends on EconomyStrength and TechInnovation
        (True, True): 0.95,  # Strong Economy & High TechInnovation → Strong Job Market
        (True, False): 0.8,  # Strong Economy & Low TechInnovation → Strong Job Market
        (False, True): 0.6,  # Weak Economy & High TechInnovation → Strong Job Market
        (False, False): 0.4  # Weak Economy & Low TechInnovation → Strong Job Market
    }),

    ('Urbanisation', 'EconomyStrength JobMarket', {  # Urbanisation depends on EconomyStrength and JobMarket
        (True, True): 0.8,  # Strong Economy & Strong Job Market → High Urbanisation
        (True, False): 0.6,  # Strong Economy & Weak Job Market → High Urbanisation
        (False, True): 0.5,  # Weak Economy & Strong Job Market → High Urbanisation
        (False, False): 0.3  # Weak Economy & Weak Job Market → High Urbanisation
    }),

    ('CleanEnergyAdoption', 'GovernmentPolicy TechInnovation',  # CleanEnergyAdoption depends on GovernmentPolicy and TechInnovation
     {
         (True, True): 0.9,  # Supportive Gov & High TechInnovation → High CleanEnergyAdoption
         (True, False): 0.6,  # Supportive Gov & Low TechInnovation → High CleanEnergyAdoption
         (False, True): 0.5,  # Unsupportive Gov & High TechInnovation → High CleanEnergyAdoption
         (False, False): 0.3  # Unsupportive Gov & Low TechInnovation → High CleanEnergyAdoption
     }),

    ('CarbonEmissions', 'Urbanisation CleanEnergyAdoption', {  # CarbonEmissions depends on Urbanisation and CleanEnergyAdoption
        (True, True): 0.5,  # High Urbanisation & High CleanEnergyAdoption → High CarbonEmissions
        (True, False): 0.8,  # High Urbanisation & Low CleanEnergyAdoption → High CarbonEmissions
        (False, True): 0.3,  # Low Urbanisation & High CleanEnergyAdoption → High CarbonEmissions
        (False, False): 0.6  # Low Urbanisation & Low CleanEnergyAdoption → High CarbonEmissions
    }),

    ('EcologicalFootprint', 'CarbonEmissions Urbanisation', {  # EcologicalFootprint depends on CarbonEmissions and Urbanisation
        (True, True): 0.9,  # High CarbonEmissions & High Urbanisation → High EcologicalFootprint
        (True, False): 0.7,  # High CarbonEmissions & Low Urbanisation → High EcologicalFootprint
        (False, True): 0.4,  # Low CarbonEmissions & High Urbanisation → High EcologicalFootprint
        (False, False): 0.2  # Low CarbonEmissions & Low Urbanisation → High EcologicalFootprint
    }),
])

def generate_graph(network):
    G = nx.DiGraph()
    for node in network.nodes:
        node_name = node.variable
        parents = node.parents
        if parents:
            for parent in parents:
                G.add_edge(parent, node_name)
        else:
            G.add_node(node_name)

    pos = {
        "EconomyStrength": (0, 5),
        "TechInnovation": (1, 4),
        "JobMarket": (1, 6),
        "Urbanisation": (2, 5),
        "GovernmentPolicy": (1, 2),
        "CleanEnergyAdoption": (3, 3),
        "CarbonEmissions": (4, 5),
        "EcologicalFootprint": (5, 5),
    }

    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_size=3000, node_color="lightblue",
        font_size=10, font_weight="bold", arrowsize=15
    )
    plt.title("Bayesian Network Graph (Custom Layout)", fontsize=14)
    plt.show()

def display_cpt_tables(network):
    for node in network.nodes:
        node_name = node.variable
        parents = node.parents
        cpt = node.cpt

        print(f"Node: {node_name}")
        if parents:
            data = []
            if isinstance(cpt, dict):  # When CPT is a dictionary
                for parent_values, prob in cpt.items():
                    row = list(parent_values) + [prob]
                    data.append(row)
                columns = list(parents) + [f"P({node_name})"]
            else:  # Case where CPT is a simpler structure (unlikely but included for safety)
                data = [[value, prob] for value, prob in cpt.items()]
                columns = [list(parents)[0], f"P({node_name})"]
            df = pd.DataFrame(data, columns=columns)
        else:  # No parents, unconditional probability
            if isinstance(cpt, dict):  # Handle cases where cpt is a dictionary
                df = pd.DataFrame([(value, prob) for value, prob in cpt.items()],
                                  columns=[node_name, "Probability"])
            else:  # Handle scalar unconditional probability
                df = pd.DataFrame([[True, cpt], [False, 1 - cpt]], columns=[node_name, "Probability"])

        print(df)
        print("\n")

# Querying the Bayesian Network
def query_bayesian_network():
    # Exact Inference: Enumeration Ask
    print("P(CarbonEmissions = High | CleanEnergyAdoption = Low, Urbanisation = High):")
    print(
        enumeration_ask('CarbonEmissions', {'CleanEnergyAdoption': False, 'Urbanisation': True}, network).show_approx())

    print("\nP(CleanEnergyAdoption = High | GovernmentPolicy = Supportive, TechInnovation = High):")
    print(enumeration_ask('CleanEnergyAdoption', {'GovernmentPolicy': True, 'TechInnovation': True},
                          network).show_approx())

    print("\nP(EcologicalFootprint = High | CarbonEmissions = High, Urbanisation = Low):")
    print(
        enumeration_ask('EcologicalFootprint', {'CarbonEmissions': True, 'Urbanisation': False}, network).show_approx())

    # Approximate Inference: Gibbs Sampling
    print("\nApproximate Inference (Gibbs Sampling) - P(CarbonEmissions = High | Urbanisation = High):")
    print(gibbs_ask('CarbonEmissions', {'Urbanisation': True}, network, N=1000).show_approx())

    print(
        "\nApproximate Inference (Likelihood Weighting) - P(CleanEnergyAdoption = High | GovernmentPolicy = Supportive):")
    print(likelihood_weighting('CleanEnergyAdoption', {'GovernmentPolicy': True}, network, N=1000).show_approx())

query_bayesian_network()
generate_graph(network)
display_cpt_tables(network)

# Question 3
print_section("QUESTION 3: NAIVE BAYES CLASSIFICATION")

class NaiveBayesClassifier:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.features = [col for col in data.columns if col != target]
        self.classes = data[target].unique()

    def get_class_data(self, cls):
        return self.data[self.data[self.target] == cls]

    def prior_probabilities(self):
        """
        Calculate prior probabilities P(C) for every class.
        """
        priors = {}
        total_count = len(self.data)
        class_counts = self.data[self.target].value_counts()

        for cls, count in class_counts.items():
            priors[cls] = count / total_count

        return priors

    def evidence_probability(self):
        """
        Estimate the probability of the evidence P(X) in the dataset:
        P(X) = 1 / (number of possible combinations of feature values)
        """
        total = 1
        for feature in self.features:
            total *= self.data[feature].nunique()
        return 1 / total

    def print_prior_probabilities(self, priors):
        rows = [{'Class': cls, 'Prior Probability': prob} for cls, prob in priors.items()]
        df = pd.DataFrame(rows)
        print("\nPrior Probabilities (P(C)):")
        print(df.to_string(index=False))

    def print_likelihoods(self, likelihoods):

        print("\nLikelihoods (P(X|C)):")

        for cls in self.classes:
            rows = []
            for feature in self.features:
                likelihood = likelihoods[cls][feature]
                for value, prob in likelihood.items():
                    rows.append({
                        'Class': cls,
                        'Feature': feature,
                        'Feature Value': value,
                        'Likelihood': prob
                    })

            # Create a DataFrame and display it
            df = pd.DataFrame(rows)
            print(f"\nClass {cls} Likelihoods:")
            print(df.to_string(index=False))

class MultinomialNaiveBayesClassifier(NaiveBayesClassifier):

    def likelihoods(self):
        """
        Calculate likelihoods P(X|C) for each class and feature.
        """
        likelihoods = {}
        for cls in self.classes:
            likelihoods[cls] = self.class_likelihood(cls)

        return likelihoods

    def class_likelihood(self, cls):
        class_data = self.get_class_data(cls)
        class_count = len(class_data)

        class_likelihoods = {}
        for feature in self.features:
            feature_count = class_data[feature].value_counts()
            class_likelihoods[feature] = self.feature_likelihood(feature, class_count, feature_count)

        return class_likelihoods

    def feature_likelihood(self, feature, class_count, feature_count):

        unique_values = self.data[feature].nunique()
        feature_likelihood = {}
        for value in self.data[feature].unique():
            count = feature_count.get(value, 0)
            # Laplace smoothing
            probability = (count + 1) / (class_count + unique_values)
            feature_likelihood[value] = probability

        return feature_likelihood

class GaussianNaiveBayesClassifier(NaiveBayesClassifier):

    def statistics(self):
        """
        Sets the mean and standard deviation for all features
        """
        stats = {}
        for cls in self.classes:
            class_data = self.get_class_data(cls)
            stats[cls] = self.class_statistics(class_data)
        return stats

    def class_statistics(self, class_data):
        stats = {}
        for feature in self.features:
            stats[feature] = self.feature_statistics(class_data, feature)
        return stats

    def feature_statistics(self, class_data, feature):
        """
        Returns the mean and standard deviation for a given feature.
        """
        return {
            'mean': class_data[feature].mean(),
            'std': class_data[feature].std()
        }

    def print_likelihoods(self, statistics):
        print("\nClass-Specific Statistics (Mean and Std by Feature):")

        for cls, features in statistics.items():
            rows = []
            for feature, stats in features.items():
                rows.append({
                    'Class': cls,
                    'Feature': feature,
                    'Mean': stats['mean'],
                    'Standard Deviation': stats['std']
                })

            df = pd.DataFrame(rows)
            print(f"\nClass {cls} Statistics:")
            print(df.to_string(index=False))

def calculate_probabilities_for_mushroom_data(mushroom_data):
    # Create the classifier
    mushroom_classifier = MultinomialNaiveBayesClassifier(mushroom_data, target="class")

    # Calculate the prior probabilities
    priors = mushroom_classifier.prior_probabilities()
    mushroom_classifier.print_prior_probabilities(priors)

    # Estimate the probability of the evidence
    evidence_probability = mushroom_classifier.evidence_probability()
    print(f"\nEstimated Probability of the Evidence (P(X)):")
    print(evidence_probability)

    # Determine the likelihood of the evidence (the numerator of Bayes’ formula)
    likelihoods = mushroom_classifier.likelihoods()
    mushroom_classifier.print_likelihoods(likelihoods)

def calculate_probabilities_for_banknote_data(banknote_data):

    banknote_classifier = GaussianNaiveBayesClassifier(banknote_data, target="class")

    # Calculate and print prior probabilities
    priors = banknote_classifier.prior_probabilities()
    banknote_classifier.print_prior_probabilities(priors)

    # Estimate the probability of the evidence
    evidence_prob = banknote_classifier.evidence_probability()
    print(f"\nEstimated Probability of the Evidence (P(X)):\n{evidence_prob:.6e}")

    # Calculate and print statistics (mean and standard deviation for each feature and class)
    statistics = banknote_classifier.statistics()
    banknote_classifier.print_likelihoods(statistics)

def load_mushroom_data():
    col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                 'stalk-surface-below-ring', 'stalk-color-above-ring',
                 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                 'ring-type', 'spore-print-color', 'population', 'habitat']

    # Load the dataset
    mushroom_data_file = "data/mushroom/agaricus-lepiota.data"
    mushroom_data = pd.read_csv(mushroom_data_file, header=None, names=col_names)

    # Encode all columns (including the target 'class')
    for col in mushroom_data.columns:
        mushroom_data[col] = mushroom_data[col].astype('category').cat.codes

    return mushroom_data

def load_banknote_data():
    data_file = "data/data_banknote_authentication.txt"
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    banknote_data = pd.read_csv(data_file, header=None, names=columns)

    # Standardize the continuous features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(banknote_data.drop(columns=["class"]))
    scaled_data = pd.DataFrame(scaled_features, columns=columns[:-1])
    scaled_data["class"] = banknote_data["class"]

    return scaled_data

def prepare_data(data, target_col='class'):
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def create_dataset(X_train, y_train, data):
    # Prepare inputs and target indices
    features = X_train.columns.tolist()
    inputs = list(range(len(features)))
    target = len(features)

    # Create examples
    examples = X_train.values.tolist()

    # Append target
    for i, row in enumerate(examples):
        row.append(y_train.values[i])

    # Create the values dictionary
    values = {
        idx: list(data[col_name].unique())
        for idx, col_name in enumerate(features)
    }

    # Add target values
    values[target] = list(data['class'].unique())

    return DataSet(name='Data', examples=examples, inputs=inputs, target=target, values=values)

def evaluate_model(model, X_test, y_test):
    # Prepare examples
    examples = X_test.values.tolist()

    y_pred = []
    for row in examples:
        y_pred.append(model(row))  # Predict using the model

    # Evaluate the classifier
    correct = sum(1 for i in range(len(y_test)) if y_pred[i] == y_test.values[i])

    # Compute accuracy
    accuracy = correct / len(y_test)

    return accuracy, y_test.values, y_pred

def show_confusion_matrix(y_true, y_pred, class_names, dataset_name="Dataset"):

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot using seaborn heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def decision_boundaries_for_all_pairs(model, X, y, feature_names, resolution=0.1):
    """
    This function was taken from the code provided by Ruairí D. O’Reilly's solution for lab 9
    Plot decision boundaries for all combinations of two features.
    """
    feature_combinations = list(combinations(range(X.shape[1]), 2))  # All 2-feature pairs
    for feature1, feature2 in feature_combinations:
        # Create grid for visualization
        x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
        y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution),
                               np.arange(y_min, y_max, resolution))

        # Fill grid data with predictions from the model
        grid_data = np.zeros((xx1.size, X.shape[1]))
        grid_data[:, feature1] = xx1.ravel()
        grid_data[:, feature2] = xx2.ravel()
        Z = np.array([model(row) for row in grid_data])
        Z = Z.reshape(xx1.shape)

        # Plot decision boundary
        plt.figure(figsize=(6, 6))
        plt.contourf(xx1, xx2, Z, alpha=0.8, cmap=ListedColormap(['red', 'blue', 'green']))
        plt.scatter(X[:, feature1], X[:, feature2], c=y, edgecolor='k', cmap=ListedColormap(['red', 'blue', 'green']))
        plt.xlabel(feature_names[feature1])
        plt.ylabel(feature_names[feature2])
        plt.title(f"Decision Boundary: {feature_names[feature1]} vs {feature_names[feature2]}")
        plt.show()

def train_data(data, dataset_name, continuous=False):
    """
    Train a Naive Bayes classifier and display visualizations for a dataset.

    Args:
        data (pd.DataFrame): The dataset to train on.
        dataset_name (str): Name of the dataset (used for labeling outputs).
        continuous (bool): Whether the dataset has continuous features.
    """
    # Step 1: Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 2: Create Dataset object
    dataset = create_dataset(X_train, y_train, data)

    # Step 3: Train the Naive Bayes classifier
    model = NaiveBayesLearner(dataset, continuous=continuous)

    # Step 4: Evaluate the classifier
    accuracy, y_true, y_pred = evaluate_model(model, X_test, y_test)

    print(f"Accuracy for {dataset_name}: {accuracy * 100:.2f}%")

    # Step 5: Plot Confusion Matrix
    class_names = [str(cls) for cls in np.unique(y_true)]
    # Remove plt.title from here
    show_confusion_matrix(y_true, y_pred, class_names=class_names, dataset_name=dataset_name)

    # Step 6: Plot Decision Boundaries (if applicable)
    if continuous:
        X_test_np = X_test.values
        y_test_np = y_test.values
        feature_names = X_test.columns.tolist()
        print(f"Plotting decision boundaries for {dataset_name}...")
        decision_boundaries_for_all_pairs(model, X_test_np, y_test_np, feature_names)

mushroom_data = load_mushroom_data()
calculate_probabilities_for_mushroom_data(mushroom_data)
train_data(mushroom_data, dataset_name="Mushroom Dataset", continuous=False)

banknote_data = load_banknote_data()
calculate_probabilities_for_banknote_data(banknote_data)
train_data(banknote_data, dataset_name="Banknote Dataset", continuous=True)



