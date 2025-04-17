import networkx as nx
from collections import defaultdict
from typing import List, Set


def louvain_algorithm(G: nx.Graph) -> List[Set]:
    """
    Apply the Louvain algorithm for community detection.
    """
    import community
    partition = community.best_partition(G)
    c_to_node = defaultdict(set)
    for k, v in partition.items():
        c_to_node[v].add(k)
    return list(c_to_node.values())
