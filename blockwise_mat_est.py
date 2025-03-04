import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.KalmanProcess import KalmanProcess
from src.EM import EMParameterEstimation
from src.GraphEM import GraphEMforA
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

seed = np.random.seed(42)

# 1. Model the question as a linear Gaussian state space model
DIM_X = 16

TRANSITION_DATA_PATH = "data/block_diagonal_A.npy"
A = np.load(TRANSITION_DATA_PATH)
Q = np.eye(DIM_X) * 0.01
H = np.eye(DIM_X)
R = np.eye(DIM_X) * 0.01
m0 = np.zeros(DIM_X)
P0 = np.eye(DIM_X) * 0.01

MODEL_PARAMS = {
    "A": A,
    "Q": Q,
    "H": H,
    "R": R,
    "m0": m0,
    "P0": P0,
}

# 2. Generate data from the model
CAR_MODEL = KalmanProcess(**MODEL_PARAMS)
DATA = CAR_MODEL.generate_measurement(T=1000)

Y = CAR_MODEL.Y
# Y = data["Y 1:T"]

# 3. Fit the model to the data
FILTER = CAR_MODEL.Filter(Y=Y)
SMOOTHER = CAR_MODEL.Smoother(Y=Y)

# 4. Plot the trajectory of the car and the estimated trajectory
# 5. Fit the estimation model to the data with missing values
NEG_LOG_LIKELIHOOD = {
    "True": -CAR_MODEL.loglikelihood(Y=Y, **MODEL_PARAMS),
    "EM": None,
    "GraphEM Laplace": None,
    "GraphEM Gaussian": None,
    "GraphEM Laplace+Gaussian": None,
}

TRANSITION_MATRIX = {
    "True": A,
    "EM": None,
    "GraphEM Laplace": None,
    "GraphEM Gaussian": None,
    "GraphEM Laplace+Gaussian": None,
}

NUM_ITERATION = 20

ALG_EM = EMParameterEstimation(**MODEL_PARAMS)
missing_vars = ["A"]
results = ALG_EM.parameter_estimation(missing_vars=missing_vars, Y=Y, num_iteration=NUM_ITERATION)

NEG_LOG_LIKELIHOOD["EM"] = results["A NegLoglikelihood"][1:]
TRANSITION_MATRIX["EM"] = results["A"][-1]

REG_LIST = ["Laplace", "Gaussian", "Laplace+Gaussian"]
ALG_GRAPHEM = GraphEMforA(**MODEL_PARAMS)
for reg in REG_LIST:

    graphEM_config = {
        "reg_type": reg,
        "num_iteration": NUM_ITERATION,
        "gamma": 1e-3,  # Douglas-Rachford control parameter
        "lambda": 50,  # penalty/prior control parameter
        "eps": 1e-5,
        "xi": 1e-5,
    }
    results = ALG_GRAPHEM.parameter_estimation(Y=Y, **graphEM_config)

    NEG_LOG_LIKELIHOOD[f"GraphEM {reg}"] = results["A NegLoglikelihood"][1:]
    TRANSITION_MATRIX[f"GraphEM {reg}"] = results["A"][-1]

# 6. Plot the negative log likelihoods for different estimation methods
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

methods = ["EM", "GraphEM Laplace", "GraphEM Gaussian", "GraphEM Laplace+Gaussian"]

for i, method in enumerate(methods):
    ax = axs[i // 2, i % 2]
    ax.plot(NEG_LOG_LIKELIHOOD[method], label=f'{method}', color="#ffb703")
    ax.axhline(y=NEG_LOG_LIKELIHOOD["True"], color='r', linestyle='--', label='$\mathcal{L}_0(\mathbf{A})$')
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel('$\mathcal{L}_0(\mathbf{A}^{(n)})$')
    ax.legend()

plt.tight_layout()
NEG_LOG_LIKELIHOOD_FIG_PATH = "final_result/blockwise_diagonal_estimation/figs/neg_log_likelihood.pdf"
os.makedirs(os.path.dirname(NEG_LOG_LIKELIHOOD_FIG_PATH), exist_ok=True)
plt.savefig(NEG_LOG_LIKELIHOOD_FIG_PATH)

logger.info(f"Saved the plot to {NEG_LOG_LIKELIHOOD_FIG_PATH}")

# 6. Plot the trajectory of the car, the estimated trajectory, and the estimated trajectory with missing values
def draw_weighted_directed_graph(adj_matrix, seed, position=None, node_size=50, node_color='#FEDA8B', 
                                 edge_color='#F4A582', edge_width_scale=5, arrow_size=5, font_size=6, 
                                 font_color='black', edge_label_color='red', title=None): 
    
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

# plt.style.use('default')
fig = plt.figure(figsize=(8, 10))
# fig.suptitle(f"{graph} graphic results with different reg terms", fontsize=16)

plt.subplot(3, 2, 1)
pos = draw_weighted_directed_graph(A, seed=seed, title="Transition Matrix", edge_color="#be95c4")

plt.subplot(3, 2, 3)
draw_weighted_directed_graph(TRANSITION_MATRIX["EM"], seed=seed, position=pos, title="EM")

for i, method in enumerate(methods):
    plt.subplot(3, 2, i+3)
    draw_weighted_directed_graph(TRANSITION_MATRIX[method], seed=seed, position=pos, title=f"{method}")

plt.tight_layout()

GRAPHIC_FIG_PATH = "final_result/blockwise_diagonal_estimation/figs/graphs_for_true_and_EM.pdf"
os.makedirs(os.path.dirname(GRAPHIC_FIG_PATH), exist_ok=True)
plt.savefig(GRAPHIC_FIG_PATH)

logger.info(f"Saved the plot to {GRAPHIC_FIG_PATH}")

# print(np.round(TRANSITION_MATRIX["EM"], 3))


