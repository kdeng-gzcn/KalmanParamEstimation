# Graph EM main
# we import class for conductin experiments
# and this main.py is for store and analyze the results

# 0. import pkg
import numpy as np
from matplotlib import pyplot as plt
from Model.GraphEM import GraphEMforA

# 1. load model
# 1.1 setting model params and hyper params
dim_x = 5
A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
H = np.eye(dim_x)  # H matrix as identity
R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
m0 = np.zeros(dim_x)  # Zero vector for mu_0
P0 = np.eye(dim_x) * 0.01  # Small values in P_0
# 1.2 load model
model = GraphEMforA(A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
# 1.3 run model and get results
"""
return {"A iterations": A_list, "Fnorm iterations": Fnorm_list, "Simple Q iterations": obj_list, "General Q iteratioins": None, "Loglikelihood iterations": None}
"""
results = model.parameter_estimation(num_iteration=1000, gamma=0.1, eps=1e-5, xi=1e-5)

# 2. analysis
# 2.1 unpack results
A_list = results["A iterations"]
Fnorm_list = results["Fnorm iterations"]
Q_list = results["Simple Q iterations"]
# 2.2 visulization
# 2.2.1 print final A
print("True A:\n", A)
print("Final A:\n", A_list[-1])
print("Final Fnorm:\n", Fnorm_list[-1])
print("Final Q:\n", Q_list[-1])
# 2.2.2 plot fnorm + obj
fig = plt.figure(figsize=(10, 6))
fig.suptitle("GraphEM Algorithm")

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(Fnorm_list, c="c")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\| A_{true} - A^{(t)} \|_{F}$")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(Q_list)
ax2.set_xlabel("t")
ax2.set_ylabel(r"$ \mathcal{Q} (A^{(t)}, A=A^{(t)}) $")

plt.tight_layout()
plt.show()

import networkx as nx

# Function to plot a weighted directed graph
def plot_weighted_directed_graph(matrix, title="Weighted Directed Graph"):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph())  # Create a directed graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, arrows=True)
    # Get edge weights and display them as labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d:.2f}" for u, v, d in G.edges(data='weight')})
    plt.title(title)
    plt.show()

# Plot the initial matrix as a directed graph
plot_weighted_directed_graph(A, title="Initial A Matrix as Weighted Directed Graph")

# Plot the final matrix as a directed graph
plot_weighted_directed_graph(A_list[-1], title="Final A Matrix as Weighted Directed Graph")
