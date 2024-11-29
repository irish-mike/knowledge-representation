import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from aima.probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, likelihood_weighting


# Define the Bayesian Network
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


if __name__ == "__main__":
    # Query the Bayesian Network
    query_bayesian_network()

    # Generate and display the Bayesian Network graph
    generate_graph(network)

    # Display the CPT tables
    display_cpt_tables(network)
