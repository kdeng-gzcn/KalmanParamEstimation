"""

To test further transition matrix, we should make other params consistent as much as possible.

1. demo default A
2. so many matrix

"""

import numpy as np
import os

import networkx as nx
import matplotlib.pyplot as plt
import random

def add_random_weights(graph):
    for u, v in graph.edges():
        graph[u][v]['weight'] = random.uniform(0, 1)
    return graph

"""

Small World Graph

"""

# Small-world graph
n, k, p = 16, 4, 0.1
small_world_graph = nx.watts_strogatz_graph(n, k, p)
small_world_graph = add_random_weights(small_world_graph)
adj_matrix = nx.to_numpy_array(small_world_graph, weight='weight')
print("Small-world Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Small-world Graph")
nx.draw(
    small_world_graph,
    with_labels=True,
    node_size=500,
    node_color='skyblue',
    font_size=10,
    font_color='black'
)
plt.show()

save = 1

if save:
    dim_x = 16
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    import os
    root = "./data/prior experiment/small world graph/"
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

"""

Scale Free Graph

"""

# Scale-free graph
n, m = 16, 2
scale_free_graph = nx.barabasi_albert_graph(n, m)
scale_free_graph = add_random_weights(scale_free_graph)
adj_matrix = nx.to_numpy_array(scale_free_graph, weight='weight')
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
plt.show()

save = 1

if save:
    dim_x = 16
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    import os
    root = "./data/prior experiment/scale free graph/"
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

"""

Bipartite Graph

"""

# Bipartite graph
n1, n2 = 8, 8
bipartite_graph = nx.complete_bipartite_graph(n1, n2)
bipartite_graph = add_random_weights(bipartite_graph)
adj_matrix = nx.to_numpy_array(bipartite_graph, weight='weight')
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

plt.show()

save = 1

if save:
    dim_x = 16
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    import os
    root = "./data/prior experiment/bipartite graph/"
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

"""

Cycle Graph

"""

# Cycle graph
n = 16
cycle_graph = nx.cycle_graph(n)
cycle_graph = add_random_weights(cycle_graph)
adj_matrix = nx.to_numpy_array(cycle_graph, weight='weight')
print("Cycle Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Cycle Graph")
nx.draw(
    cycle_graph,
    with_labels=True,
    font_size=10,
    font_color='black'
)

plt.show()

save = 1

if save:
    dim_x = 16
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    import os
    root = "./data/prior experiment/cycle graph/"
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0

"""

Star Graph

"""

# Star graph
n = 16
star_graph = nx.star_graph(n - 1)
star_graph = add_random_weights(star_graph)
adj_matrix = nx.to_numpy_array(star_graph, weight='weight')
print("Star Graph Adjacency Matrix with Weights:")
print(adj_matrix)
plt.title("Star Graph")
nx.draw(
    star_graph,
    with_labels=True,
    font_color='black'
)

plt.show()

save = 1

if save:
    dim_x = 16
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    import os
    root = "./data/prior experiment/star graph/"
    os.makedirs(root, exist_ok=True)

    # Save the matrices to files
    np.save(os.path.join(root, "A.npy"), adj_matrix)  # Save matrix A
    np.save(os.path.join(root, "Q.npy"), Q)  # Save matrix Q
    np.save(os.path.join(root, "H.npy"), H)  # Save matrix H
    np.save(os.path.join(root, "R.npy"), R)  # Save matrix R
    np.save(os.path.join(root, "m0.npy"), m0)  # Save vector m0
    np.save(os.path.join(root, "P0.npy"), P0)  # Save matrix P0
