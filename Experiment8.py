# # Graph EM main with loglikelihood and EM without L1 norm penalty
# Estimate Q but with numerical problem
# and this main.py is for store and analyze the results

# 0. import pkg
import numpy as np
from matplotlib import pyplot as plt
from Model.GraphEM import GraphEMforQ
from Model.KalmanClass import EMParameterEstimationAll

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
model_MLE = EMParameterEstimationAll(var="Q", A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
model = GraphEMforQ(A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
model.Y = model_MLE.Y
model.kf = model_MLE.kf
# 1.3 run model and get results
"""
return {"A iterations": A_list, "Fnorm iterations": Fnorm_list, "Simple Q iterations": obj_list, "General Q iteratioins": None, "Loglikelihood iterations": None}
"""
results = model.parameter_estimation(num_iteration=100, gamma=0.1, eps=1e-5, xi=1e-5)
_, Q_list_MLE, Fnorm_list_MLE, Neg_Loglikelihood_list_MLE = model_MLE.parameter_estimation(num_iteration=100)

# 2. analysis
# 2.1 unpack results
Q_list = results["Q iterations"]
Fnorm_list = results["Fnorm iterations"]
Q_obj_list = results["Simple Q iterations"]
Neg_Loglikelihood_list = results["Loglikelihood iterations"]
# 2.2 visulization
# 2.2.1 print final A
print("True Q:\n", Q)
print("Final Q:\n", Q_list[-1])
print("Final Q MLE:\n", Q_list_MLE[-1])
print("Final Fnorm:\n", Fnorm_list[-1])
# print("Final Q obj function:\n", Q_obj_list[-1])
print("Final Neg Loglikelihood:\n", Neg_Loglikelihood_list[-1])
print("Final Neg Loglikelihood MLE:\n", Neg_Loglikelihood_list_MLE[-1])
# 2.2.2 plot fnorm + obj + loglikelihood
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(10, 4))
fig.suptitle("GraphEM Algorithm")

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(Fnorm_list, c=colors[0], label="GraphEM")
ax1.plot(Fnorm_list_MLE, c=colors[1], label="MLE")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\| Q_{true} - Q^{(t)} \|_{F}$")
ax1.legend()

# ax2 = fig.add_subplot(1, 3, 2)
# ax2.plot(Q_list, c=colors[1])
# ax2.set_xlabel("t")
# ax2.set_ylabel(r"$ \mathcal{Q} (A^{(t)}, A=A^{(t)}) $")

baseline = model.loglikelihood(theta=Q, Y=model.Y)

ax3 = fig.add_subplot(1, 2, 2)
ax3.plot(Neg_Loglikelihood_list, c=colors[2], label="GraphEM")
ax3.plot(Neg_Loglikelihood_list_MLE, c=colors[3], label="MLE")
ax3.axhline(y=baseline, color=colors[4], linestyle='--', label=r'$ -\ell (Q^{ture} \mid Y) $')
ax3.set_xlabel("t")
ax3.set_ylabel(r"$ -\ell (Q^{(t)} \mid Y) $")
ax3.legend()

plt.tight_layout()
plt.show()

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
