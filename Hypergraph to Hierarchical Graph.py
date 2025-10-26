import hypernetx as hnx
import networkx as nx
import torch
import numpy as np
from typing import List, Tuple, Dict

def convert_hypergraph_to_hierarchical_graph_with_embeddings(
    H: hnx.Hypergraph
) -> nx.Graph:
    """
    Convert a hypergraph (with embeddings) into a hierarchical NetworkX graph.

    Parameters
    ----------
    H : hnx.Hypergraph
        Hypergraph containing 'feature' tensors on nodes.

    Returns
    -------
    G : nx.Graph
        Hierarchical NetworkX graph with document → sentence → word structure.
        Node embeddings preserved in G.nodes[n]['feature'].
    """
    G = nx.Graph()

    # Add all nodes with their embeddings
    for n, attrs in H.nodes.items():
        G.add_node(n, feature=attrs.get('feature', torch.zeros_like(next(iter(H.nodes.values())).get('feature'))))

    # Add document → sentence edges
    for hedge_name in H.incidence_dict:
        if hedge_name.startswith("sentence_"):
            G.add_edge("document_0", hedge_name)

            # Connect each word to its sentence
            for word in H.incidence_dict[hedge_name]:
                G.add_edge(hedge_name, word)

    return G
