"""

this experiment is for testing 3 different priors for 5+ types of matrix

For each type, we find the most suitable prior for it.

"""

# 0. import pkg
import numpy as np
import os
from matplotlib import pyplot as plt
import networkx as nx

from Model.GraphEM import GraphEMforA
from Model.EM import EMParameterEstimationAll

seed = np.random.seed(0)

# 1. load model
# 1.1 setting model params and hyper params
root = "./data/prior experiment/block diagonal graph"
# root = "./data/prior experiment/scale free graph"
# root = "./data/prior experiment/bipartite graph"
graph = "Block Diagonal"
A = np.load(os.path.join(root, "A.npy"))
Q = np.load(os.path.join(root, "Q.npy"))
H = np.load(os.path.join(root, "H.npy"))
R = np.load(os.path.join(root, "R.npy"))
m0 = np.load(os.path.join(root, "m0.npy"))
P0 = np.load(os.path.join(root, "P0.npy"))

dimx = len(A)

# 1.2 load model
model_MLE = EMParameterEstimationAll(var="A", A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
model_list = []
for reg in ["Laplace", "Gaussian", "Laplace_Gaussian"]:
    model = GraphEMforA(A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0, reg_name=reg)
    model.Y = model_MLE.Y
    model_list.append(model)
# 1.3 run model and get results
"""
return {"A iterations": A_list, "Fnorm iterations": Fnorm_list, "Simple Q iterations": obj_list, "General Q iteratioins": None, "Loglikelihood iterations": None}
"""
_, A_list_MLE, Fnorm_list_MLE, Neg_Loglikelihood_list_MLE = model_MLE.parameter_estimation(num_iteration=30)
res_list = []
for idx, reg in enumerate(["Laplace", "Gaussian", "Laplace_Gaussian"]):
    results = model_list[idx].parameter_estimation(num_iteration=30, gamma=0.1, eps=1e-3, xi=1e-3)
    res_list.append(results)

# 2. analysis
# 2.1 unpack results
A_seq_list = []
Fnorm_seq_list = []
Q_seq_list = []
Neg_Loglikelihood_seq_list = []
for idx, reg in enumerate(["Laplace", "Gaussian", "Laplace_Gaussian"]):
    
    A_list, Fnorm_list, Q_list, _, Neg_Loglikelihood_list = res_list[idx].values()

    A_seq_list.append(A_list)
    Fnorm_seq_list.append(Fnorm_list)
    Q_seq_list.append(Q_list)
    Neg_Loglikelihood_seq_list.append(Neg_Loglikelihood_list)

# 2.1.1 text result
print("True A (3r, 3c):")
print(A[:3, :3])

print("Ahat from EM (3r, 3c):")
print(A_list_MLE[-1][:3, :3])

baseline = -model.loglikelihood(theta=A, Y=model.Y)

print("Baseline:")
print(baseline)

print("Final Neg Loglikelihood EM (MLEM):")
print(Neg_Loglikelihood_list_MLE[-1])

print("Final Neg Loglikelihood Laplace Reg:")
print(Neg_Loglikelihood_seq_list[0][-1])

print("Final Neg Loglikelihood Gaussian Reg:")
print(Neg_Loglikelihood_seq_list[1][-1])

print("Final Neg Loglikelihood Laplace+Gaussian Reg:")
print(Neg_Loglikelihood_seq_list[2][-1])

# 2.2 visulization
# 2.2.2 plot fnorm / obj (Q = q + reg) / loglikelihood
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"{graph} for GraphEM with different reg term")

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(Neg_Loglikelihood_list_MLE, c=colors[0], label="MLEM")
ax1.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax1.set_xlabel("t")
ax1.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax1.legend()

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(Neg_Loglikelihood_seq_list[0], c=colors[2], label="GraphEM with Laplace reg")
ax2.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax2.set_xlabel("t")
ax2.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(Neg_Loglikelihood_seq_list[1], c=colors[3], label="GraphEM with Gaussian reg")
ax3.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax3.set_xlabel("t")
ax3.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(Neg_Loglikelihood_seq_list[2], c=colors[4], label="GraphEM with Gaussian + Laplace reg")
ax4.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax4.set_xlabel("t")
ax4.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax4.legend()

plt.tight_layout()

os.makedirs("./Result/Experiment9/", exist_ok=True)
plt.savefig(f"./Result/Experiment9/{graph} with GraphEM With 3 Different Reg Term.pdf")

plt.show()

def draw_weighted_directed_graph(adj_matrix, position=None, node_size=50, node_color='#FEDA8B', 
                                 edge_color='#F4A582', edge_width_scale=7, arrow_size=10, font_size=6, 
                                 font_color='black', edge_label_color='red', title=None, seed=seed): 
    
    # Convert adjacency matrix to a directed graph
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Compute node positions if not provided
    if position is None:
        position = nx.spring_layout(graph, seed=seed)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=node_size,
        node_color=node_color,
    )

    # Draw edges with custom arrow style
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    nx.draw_networkx_edges(
        graph,
        position,
        edge_color=edge_color,
        width=[w * edge_width_scale for w in weights],
        arrowstyle='->',
        arrowsize=arrow_size,
    )

    # # Draw edge labels (weights)
    # edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()}
    # nx.draw_networkx_edge_labels(
    #     graph,
    #     position,
    #     edge_labels=edge_labels,
    #     font_color=edge_label_color,
    # )

    # Draw node labels
    nx.draw_networkx_labels(
        graph,
        position,
        font_size=font_size,
        font_color=font_color,
    )

    plt.title(title)

    return position

plt.style.use('default')
fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"{graph} graphic results with different reg terms", fontsize=16)

plt.subplot(2, 3, 1)
pos = draw_weighted_directed_graph(A, title="True Value")

plt.subplot(2, 3, 2)
draw_weighted_directed_graph(A_list_MLE[-1], position=pos, title="MLEM")

for i, (A_seq, reg) in enumerate(zip(A_seq_list, ["Laplace", "Gaussian", "Laplace_Gaussian"])):
    plt.subplot(2, 3, i + 3)
    draw_weighted_directed_graph(A_seq[-1], position=pos, title=reg)

plt.tight_layout()

os.makedirs("./Result/Experiment9/", exist_ok=True)
plt.savefig(f"./Result/Experiment9/{graph} With 3 Different Reg Graphic Results.pdf")

plt.show()
