import os
import random
import logging

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

seed = np.random.seed(42)

def _add_random_weights(graph):

    for u, v in graph.edges():

        graph[u][v]['weight'] = random.uniform(0, 1)

    return graph


RESULT_DIR = "final_result/graph_types/data"

IF_SAVE = {
    "Small World": 1,
    "Scale Free": 1,
    "Bipartite": 1,
    "Cycle": 1,
    "Star": 1,
    "Block Diagonal": 0,
}

DIM_N = 16

def _graph2array(graph_nx, graph_type: str, if_save: bool):
    adj_matrix = nx.to_numpy_array(graph_nx, weight='weight')
    U, S, VT = np.linalg.svd(adj_matrix)
    max_singular_value = np.max(S)
    coef = 0.99 / max_singular_value
    adj_matrix = coef * adj_matrix

    if if_save:
        root = os.path.join(RESULT_DIR, graph_type)
        os.makedirs(root, exist_ok=True)
        np.save(os.path.join(root, "A_raw.npy"), adj_matrix)


    adj_matrix = nx.to_numpy_array(graph_nx, weight='weight')

    U, S, VT = np.linalg.svd(adj_matrix)
    S = np.minimum(S, 0.99)
    adj_matrix = U @ np.diag(S) @ VT

    # weights = [graph_nx[u][v]['weight'] for u, v in graph_nx.edges()]

    logger.info(f"{graph_type} Matrix: ")
    print(adj_matrix)

    if if_save:
        dim_x = DIM_N
        Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
        H = np.eye(dim_x)  # H matrix as identity
        R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
        m0 = np.zeros(dim_x)  # Zero vector for mu_0
        P0 = np.eye(dim_x) * 0.01  # Small values in P_0

        root = os.path.join(RESULT_DIR, graph_type)
        os.makedirs(root, exist_ok=True)

        # Save the matrices to files
        np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
        np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
        np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
        np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
        np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
        np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0`

"""

Small World Graph

"""

# Small-world graph
k, p = 4, 0.3
small_world_graph = nx.watts_strogatz_graph(DIM_N, k, p)
small_world_graph = _add_random_weights(small_world_graph)

graph_type = "Small World"
_graph2array(small_world_graph, graph_type, if_save=IF_SAVE[graph_type])

"""

Scale Free Graph

"""

# Scale-free graph
m = 2
scale_free_graph = nx.barabasi_albert_graph(DIM_N, m)
scale_free_graph = _add_random_weights(scale_free_graph)

graph_type = "Scale Free"
_graph2array(scale_free_graph, graph_type, if_save=IF_SAVE[graph_type])

"""

Bipartite Graph

"""

# Bipartite graph
n1, n2 = int(DIM_N/2), int(DIM_N/2)
bipartite_graph = nx.complete_bipartite_graph(n1, n2)
bipartite_graph = _add_random_weights(bipartite_graph)

graph_type = "Bipartite"
_graph2array(bipartite_graph, graph_type, if_save=IF_SAVE[graph_type])

"""

Cycle Graph

"""

# Cycle graph
cycle_graph = nx.cycle_graph(DIM_N)
cycle_graph = _add_random_weights(cycle_graph)

graph_type = "Cycle"
_graph2array(cycle_graph, graph_type, if_save=IF_SAVE[graph_type])


"""

Star Graph

"""

# Star graph
star_graph = nx.star_graph(DIM_N-1)
star_graph = _add_random_weights(star_graph)

graph_type = "Star"
_graph2array(star_graph, graph_type, if_save=IF_SAVE[graph_type])


"""

Block Diagonal

"""

# # Block Diagonal Graph
# block_sizes = [4, 4, 4, 4]  # Define the sizes of the blocks
# block_diagonal_graph = nx.DiGraph()

# # Create each block as a complete graph and add it to the block diagonal graph
# start_node = 0
# for size in block_sizes:
#     block_graph = nx.complete_graph(size, create_using=nx.DiGraph())
#     block_graph = _add_random_weights(block_graph)  # Add random weights to the block
#     block_diagonal_graph = nx.disjoint_union(block_diagonal_graph, block_graph)
#     start_node += size

# # Convert the graph to an adjacency matrix
# adj_matrix_block = nx.to_numpy_array(block_diagonal_graph, weight='weight')

# # Normalize the adjacency matrix using SVD (optional)
# U, S, VT = np.linalg.svd(adj_matrix_block)
# max_singular_value = np.max(S)
# coef = 0.99 / max_singular_value
# adj_matrix_block = coef * adj_matrix_block

# print("Block Diagonal Graph Adjacency Matrix with Weights:")
# print(adj_matrix_block)

# plt.figure(figsize=(10, 8))
# plt.title("Block Diagonal Graph with Directed Edges")

# # Extract edge weights for visualization
# weights = [block_diagonal_graph[u][v]['weight'] for u, v in block_diagonal_graph.edges()]

# # Compute node positions if not provided
# position = nx.spring_layout(block_diagonal_graph, seed=42)

# # Draw nodes
# nx.draw_networkx_nodes(
#     block_diagonal_graph,
#     position,
#     node_size=50,
#     node_color='#FEDA8B',
# )

# # Draw edges with custom arrow style
# weights = [block_diagonal_graph[u][v]['weight'] for u, v in block_diagonal_graph.edges()]
# nx.draw_networkx_edges(
#     block_diagonal_graph,
#     position,
#     edge_color='#F4A582',
#     width=[w * 3 for w in weights],  # Scale edge widths based on weights
#     arrowstyle='->',
#     arrowsize=10,
# )

# # Draw node labels
# nx.draw_networkx_labels(
#     block_diagonal_graph,
#     position,
#     font_size=6,
#     font_color='black',
# )

# if IF_SAVE["Block Diagonal"]:

#     # plt.show()

#     # Define dimensions and matrices for saving
#     dim_x = DIM_N
#     Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
#     H = np.eye(dim_x)  # H matrix as identity
#     R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
#     m0 = np.zeros(dim_x)  # Zero vector for mu_0
#     P0 = np.eye(dim_x) * 0.01  # Small values in P_0

#     # Create directory for saving results
#     root = os.path.join(RESULT_DIR, "block diagonal graph/")
#     os.makedirs(root, exist_ok=True)

#     # Save the matrices to files
#     np.save(os.path.join(root, "A.npy"), adj_matrix_block)  # Save matrix A
#     np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
#     np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
#     np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
#     np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
#     np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

# plt.close()
