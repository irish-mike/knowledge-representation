# Ensure the correct path to aima-python is set
import sys
sys.path.append('./aima-python')  # Adjust the path if necessary

from aima.search import UndirectedGraph, GraphProblem, compare_searchers
from aima.search import breadth_first_tree_search, breadth_first_graph_search
from aima.search import depth_first_tree_search, depth_first_graph_search
from aima.search import depth_limited_search, iterative_deepening_search

# Updated Ireland map with interconnectivity
ireland_map = UndirectedGraph({
    'Dublin': {'Navan': 50, 'Drogheda': 55, 'Mullingar': 85, 'Kilkenny': 120},
    'Navan': {'Dublin': 50, 'Kells': 20, 'Trim': 15},
    'Kells': {'Navan': 20, 'Monaghan': 70},
    'Trim': {'Navan': 15, 'Carlow': 60},
    'Carlow': {'Trim': 60, 'Kilkenny': 30},
    'Kilkenny': {'Dublin': 120, 'Carlow': 30, 'Waterford': 60},
    'Waterford': {'Kilkenny': 60, 'Cork': 120},
    'Cork': {'Waterford': 120, 'Limerick': 90},
    'Limerick': {'Cork': 90, 'Galway': 90},
    'Galway': {'Limerick': 90, 'Sligo': 100, 'Athlone': 80},
    'Sligo': {'Galway': 100, 'Donegal': 80},
    'Donegal': {'Sligo': 80, 'Monaghan': 90},
    'Monaghan': {'Kells': 70, 'Donegal': 90, 'Belfast': 75},
    'Belfast': {'Monaghan': 75, 'Newry': 60},
    'Newry': {'Belfast': 60, 'Dundalk': 20},
    'Dundalk': {'Newry': 20, 'Drogheda': 35},
    'Drogheda': {'Dundalk': 35, 'Dublin': 55},
    'Mullingar': {'Dublin': 85, 'Athlone': 50},
    'Athlone': {'Mullingar': 50, 'Galway': 80}
})

# Define the problems using updated towns
problem1 = GraphProblem('Dublin', 'Donegal', ireland_map)
problem2 = GraphProblem('Cork', 'Belfast', ireland_map)
problem3 = GraphProblem('Navan', 'Galway', ireland_map)

# Define search algorithms to compare
searchers = [
    breadth_first_tree_search,
    breadth_first_graph_search,
    #depth_first_tree_search,
    depth_first_graph_search,
    iterative_deepening_search,
    lambda p: depth_limited_search(p, limit=10)  # Set depth limit
]

# Compare the search algorithms on the defined problems
compare_searchers(
    problems=[problem1, problem2, problem3],
    header=['Searcher', 'Dublin to Donegal', 'Cork to Belfast', 'Navan to Galway'],
    searchers=searchers
)
