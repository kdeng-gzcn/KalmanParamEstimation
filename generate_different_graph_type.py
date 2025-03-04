import os
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def add_random_weights(graph):

    for u, v in graph.edges():

        graph[u][v]['weight'] = random.uniform(0, 1)

    return graph

result_dir = "./data/prior experiment/"

save = {
    "Small World": 0,
    "Scale Free": 0,
    "Bipartite": 0,
    "Cycle": 0,
    "Star": 0,
    "Block Diagonal": 1,
}

n = 16

"""

Small World Graph

"""

# Small-world graph
k, p = 4, 0.1
small_world_graph = nx.watts_strogatz_graph(n, k, p)
small_world_graph = add_random_weights(small_world_graph)
adj_matrix = nx.to_numpy_array(small_world_graph, weight='weight')

U, S, VT = np.linalg.svd(adj_matrix)
max_singular_value = np.max(S)
coef = 0.99 / max_singular_value
adj_matrix = coef * adj_matrix

print("Small-world Graph Adjacency Matrix with Weights:")
print(adj_matrix)

weights = [small_world_graph[u][v]['weight'] for u, v in small_world_graph.edges()]

plt.figure(figsize=(10, 8))
plt.title("Small-world Graph with Weighted Edges", fontsize=16)

nx.draw(
    small_world_graph,
    with_labels=True,
    node_size=500,
    node_color='skyblue',
    font_size=10,
    font_color='black',
    width=[w * 5 for w in weights],
    edge_color='blue',
    pos=nx.spring_layout(small_world_graph, seed=42),
)

if save["Small World"]:

    plt.show()

    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    root = os.path.join(result_dir, "small world graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()

"""

Scale Free Graph

"""

# Scale-free graph
m = 2
scale_free_graph = nx.barabasi_albert_graph(n, m)
scale_free_graph = add_random_weights(scale_free_graph)
adj_matrix = nx.to_numpy_array(scale_free_graph, weight='weight')

# U, S, VT = np.linalg.svd(adj_matrix)
# max_singular_value = np.max(S)
# coef = 0.99 / max_singular_value
# adj_matrix = coef * adj_matrix

print("Scale-free Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Scale-free Graph")
nx.draw(
    scale_free_graph,
    with_labels=True,
    node_size=500,
    node_color='skyblue',
    font_size=10,
    font_color='black'
)

if save["Scale Free"]:

    plt.show()

    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    root = os.path.join(result_dir, "scale free graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()

"""

Bipartite Graph

"""

# Bipartite graph
n1, n2 = int(n/2), int(n/2)
bipartite_graph = nx.complete_bipartite_graph(n1, n2)
bipartite_graph = add_random_weights(bipartite_graph)
adj_matrix = nx.to_numpy_array(bipartite_graph, weight='weight')

# U, S, VT = np.linalg.svd(adj_matrix)
# max_singular_value = np.max(S)
# coef = 0.99 / max_singular_value
# adj_matrix = coef * adj_matrix

print("Bipartite Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Bipartite Graph")
nx.draw(
    bipartite_graph,
    with_labels=True,
    node_color='skyblue',
    font_size=10,
    font_color='black'
)

if save["Bipartite"]:

    plt.show()

    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    root = os.path.join(result_dir, "bipartite graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()

"""

Cycle Graph

"""

# Cycle graph
cycle_graph = nx.cycle_graph(n)
cycle_graph = add_random_weights(cycle_graph)
adj_matrix = nx.to_numpy_array(cycle_graph, weight='weight')

U, S, VT = np.linalg.svd(adj_matrix)
max_singular_value = np.max(S)
coef = 0.99 / max_singular_value
adj_matrix = coef * adj_matrix

print("Cycle Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Cycle Graph")
nx.draw(
    cycle_graph,
    with_labels=True,
    font_size=10,
    font_color='black'
)

if save["Cycle"]:

    plt.show()

    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    root = os.path.join(result_dir, "cycle graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()

"""

Star Graph

"""

# Star graph
star_graph = nx.star_graph(n - 1)
star_graph = add_random_weights(star_graph)
adj_matrix = nx.to_numpy_array(star_graph, weight='weight')

# U, S, VT = np.linalg.svd(adj_matrix)
# max_singular_value = np.max(S)
# coef = 0.99 / max_singular_value
# adj_matrix = coef * adj_matrix

print("Star Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Star Graph")
nx.draw(
    star_graph,
    with_labels=True,
    font_color='black'
)

if save["Star"]:

    plt.show()

    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    root = os.path.join(result_dir, "star graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()

"""

Block Diagonal

"""

# Block Diagonal Graph
block_sizes = [4, 4, 4, 4]  # Define the sizes of the blocks
block_diagonal_graph = nx.DiGraph()

# Create each block as a complete graph and add it to the block diagonal graph
start_node = 0
for size in block_sizes:
    block_graph = nx.complete_graph(size, create_using=nx.DiGraph())
    block_graph = add_random_weights(block_graph)  # Add random weights to the block
    block_diagonal_graph = nx.disjoint_union(block_diagonal_graph, block_graph)
    start_node += size

# Convert the graph to an adjacency matrix
adj_matrix_block = nx.to_numpy_array(block_diagonal_graph, weight='weight')

# Normalize the adjacency matrix using SVD (optional)
U, S, VT = np.linalg.svd(adj_matrix_block)
max_singular_value = np.max(S)
coef = 0.99 / max_singular_value
adj_matrix_block = coef * adj_matrix_block

print("Block Diagonal Graph Adjacency Matrix with Weights:")
print(adj_matrix_block)

plt.figure(figsize=(10, 8))
plt.title("Block Diagonal Graph with Directed Edges")

# Extract edge weights for visualization
weights = [block_diagonal_graph[u][v]['weight'] for u, v in block_diagonal_graph.edges()]

# Compute node positions if not provided
position = nx.spring_layout(block_diagonal_graph, seed=42)

# Draw nodes
nx.draw_networkx_nodes(
    block_diagonal_graph,
    position,
    node_size=50,
    node_color='#FEDA8B',
)

# Draw edges with custom arrow style
weights = [block_diagonal_graph[u][v]['weight'] for u, v in block_diagonal_graph.edges()]
nx.draw_networkx_edges(
    block_diagonal_graph,
    position,
    edge_color='#F4A582',
    width=[w * 3 for w in weights],  # Scale edge widths based on weights
    arrowstyle='->',
    arrowsize=10,
)

# Draw node labels
nx.draw_networkx_labels(
    block_diagonal_graph,
    position,
    font_size=6,
    font_color='black',
)

if save["Block Diagonal"]:

    plt.show()

    # Define dimensions and matrices for saving
    dim_x = n
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    # Create directory for saving results
    root = os.path.join(result_dir, "block diagonal graph/")
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix_block)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

plt.close()
