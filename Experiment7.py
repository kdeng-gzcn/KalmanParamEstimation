"""

GraphEM for A for 3 different prior, EM for A, for comparasion

"""
# # Compare Graph EM with loglikelihood and EM, i.e. without L1 norm penalty
# This is for A
# and this main.py is for store and analyze the results

# 0. import pkg
import numpy as np
np.random.seed(0) # set seed 0
from matplotlib import pyplot as plt
from Model.GraphEM import GraphEMforA
from Model.KalmanClass import EMParameterEstimationAll

# 1. load model
# 1.1 setting model params and hyper params
dim_x = 16
A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
H = np.eye(dim_x)  # H matrix as identity
R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
m0 = np.zeros(dim_x)  # Zero vector for mu_0
P0 = np.eye(dim_x) * 0.01  # Small values in P_0
# 1.2 load model
model_MLE = EMParameterEstimationAll(var="A", A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
model = GraphEMforA(A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0, reg_name="Laplace")
model.Y = model_MLE.Y
# 1.3 run model and get results
"""
return {"A iterations": A_list, 
"Fnorm iterations": Fnorm_list, 
"Simple Q iterations": obj_list, 
"General Q iteratioins": None, 
"Loglikelihood iterations": None}
"""
results = model.parameter_estimation(num_iteration=30, gamma=0.1, eps=1e-5, xi=1e-5)
_, A_list_MLE, Fnorm_list_MLE, Neg_Loglikelihood_list_MLE = model_MLE.parameter_estimation(num_iteration=30)

# 2. analysis
# 2.1 unpack results
A_list = results["A iterations"]
Fnorm_list = results["Fnorm iterations"]
Q_list = results["Simple Q iterations"]
Neg_Loglikelihood_list = results["Loglikelihood iterations"]
# 2.2 visulization
# 2.2.1 print final A
print("True A:\n", A)
print("Final A:\n", A_list[-1])
print("Final A MLE:\n", A_list_MLE[-1])
print("Final Fnorm:\n", Fnorm_list[-1])
print("Final Q:\n", Q_list[-1])
print("Final Neg Loglikelihood:\n", Neg_Loglikelihood_list[-1])
print("Final Neg Loglikelihood MLE:\n", Neg_Loglikelihood_list_MLE[-1])
# 2.2.2 plot fnorm + obj + loglikelihood
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(10, 8))
fig.suptitle("GraphEM Algorithm")

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(Fnorm_list[1:], c=colors[0], label="GraphEM")
ax1.plot(Fnorm_list_MLE[1:], c=colors[1], label="MLE")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\| A_{true} - A^{(t)} \|_{F}$")
ax1.legend()

# ax2 = fig.add_subplot(1, 3, 2)
# ax2.plot(Q_list, c=colors[1])
# ax2.set_xlabel("t")
# ax2.set_ylabel(r"$ \mathcal{Q} (A^{(t)}, A=A^{(t)}) $")

baseline = -model.loglikelihood(theta=A, Y=model.Y)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(Neg_Loglikelihood_list, c=colors[2], label="GraphEM")
ax2.plot(Neg_Loglikelihood_list_MLE, c=colors[3], label="MLE")
ax2.axhline(y=baseline, color=colors[4], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax2.set_xlabel("t")
ax2.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot([a - b for a, b in zip(Neg_Loglikelihood_list_MLE, Neg_Loglikelihood_list)][1:], c=colors[5], label="MLE - GraphEM")
ax3.set_xlabel("t")
ax3.set_ylabel(r"$ \Delta -\ell (A^{(t)} \mid Y) $")
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(Q_list, c=colors[6], label="$ Q = q + L1 $")
ax4.set_xlabel("t")
ax4.set_ylabel(r"$ Q $")
ax4.legend()

plt.tight_layout()
import os
os.makedirs("./Result/Experiment7", exist_ok=True)
plt.savefig("./Result/Experiment7/GraphEM Comparasion Results.pdf")
plt.show()

"""
No Use, for plotting the weighted graph
"""
# import networkx as nx

# # Function to plot a weighted directed graph
# def plot_weighted_directed_graph(matrix, title="Weighted Directed Graph"):
#     G = nx.from_numpy_array(matrix, create_using=nx.DiGraph())  # Create a directed graph
#     pos = nx.spring_layout(G)  # Layout for positioning nodes

#     # Draw the graph
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, arrows=True)
#     # Get edge weights and display them as labels
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d:.2f}" for u, v, d in G.edges(data='weight')})
#     plt.title(title)
#     plt.show()

# # Plot the initial matrix as a directed graph
# plot_weighted_directed_graph(A, title="Initial A Matrix as Weighted Directed Graph")

# # Plot the final matrix as a directed graph
# plot_weighted_directed_graph(A_list[-1], title="Final A Matrix as Weighted Directed Graph")
