"""Workshop solutions"""

# pixi add graphviz
# pixi add --pypi networkx pygraphviz
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import phylo2vec as p2v


def adjacency_matrix(v: np.ndarray) -> np.ndarray:
    """Create the adjacency matrix of the tree represented by a phylo2vec vector.

    Parameters
    ----------
    v: np.ndarray
        A phylo2vec vector.

    Returns
    -------
    np.ndarray
        The adjacency matrix of the tree represented by the vector.
    """
    k = len(v)
    n_nodes = 2 * k + 1
    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)

    edges = p2v.to_edges(v)

    for child, parent in edges:
        adj[child, parent] = 1
    return adj


def visualize_tree(vector_or_matrix: np.ndarray):
    """Visualize the tree represented by a phylo2vec vector.

    Parameters
    ----------
    v_or_matrix:: np.ndarray
        A phylo2vec vector or matrix.
    """
    # Create a graph from the edge list
    G = nx.DiGraph()
    if vector_or_matrix.ndim == 1:
        edges = p2v.to_edges(vector_or_matrix)
        G.add_edges_from([(p, c) for c, p in edges])
    elif vector_or_matrix.ndim == 2:
        ancestry = p2v.to_ancestry(vector_or_matrix[:, 0].astype(int))
        edges = []
        for i, (c1, c2, p) in enumerate(ancestry):
            G.add_edge(p, c1, minlen=10 * vector_or_matrix[i, 1])
            G.add_edge(p, c2, minlen=10 * vector_or_matrix[i, 2])
    else:
        raise ValueError("Input must be a vector or a 2D matrix.")

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(4, 3))  # smaller than default
    nx.draw(G, pos, with_labels=True)
    plt.show()
