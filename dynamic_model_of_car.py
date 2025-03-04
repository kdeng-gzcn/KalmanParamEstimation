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
DIM_X = 4

dt = 1 # Discretization step
q1_c = 0.1  # Process noise parameter for x1
q2_c = 0.1  # Process noise parameter for x2

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]) # Transition matrix A (from Equation 4.19)

# path = "data/block_diagonal_A.npy"
# np.save(path, A)

Q = np.array([
    [(q1_c * dt**3) / 3, 0, (q1_c * dt**2) / 2, 0],
    [0, (q2_c * dt**3) / 3, 0, (q2_c * dt**2) / 2],
    [(q1_c * dt**2) / 2, 0, q1_c * dt, 0],
    [0, (q2_c * dt**2) / 2, 0, q2_c * dt]
]) # Process noise covariance matrix Q (from Equation 4.20)

# Q = np.eye(DIM_X) * 0.1  # Small noise in Sigma_q

v0_x = 10
v0_y = 10
m0 = np.array([0, 0, v0_x, v0_y]) # (x, y, vx, vy)
# m0 = np.zeros(DIM_X)

H = np.eye(DIM_X)
R = np.eye(DIM_X) * 0.01
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
DATA = CAR_MODEL.generate_measurement(T=100)

Y = CAR_MODEL.Y
# Y = data["Y 1:T"]

# 3. Fit the model to the data
FILTER = CAR_MODEL.Filter(Y=Y)
SMOOTHER = CAR_MODEL.Smoother(Y=Y)

# 4. Plot the trajectory of the car and the estimated trajectory
# Extract the true and estimated positions
_ = Y[:, :2]
X_AXIS = _[:, 0]
Y_AXIS = _[:, 1]

_ = FILTER["m 0:T"][:, :2]
FILTER_X_AXIS = _[1:, 0]
FILTER_Y_AXIS = _[1:, 1]

_ = SMOOTHER["m smo 0:T"][:, :2]
SMOOTHER_X_AXIS = _[1:, 0]
SMOOTHER_Y_AXIS = _[1:, 1]

# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(f'$T={len(Y)}, \Delta t={dt}, q_1^c = {q1_c}, q_2^c = {q2_c}, v_0^x = {v0_x}, v_0^y = {v0_y}$')

# Plot the true trajectory
axs[0, 0].scatter(X_AXIS, Y_AXIS, label='True Trajectory', color='#ffbe0b', marker='o', s=10)
axs[0, 0].set_xlabel('X Position')
axs[0, 0].set_ylabel('Y Position')
axs[0, 0].legend()
# axs[0, 0].set_title('True Trajectory')

# Plot the estimated trajectory from the filter
axs[0, 1].scatter(FILTER_X_AXIS, FILTER_Y_AXIS, label='Filtered Trajectory', color='#ff006e', marker='x', s=10)
axs[0, 1].set_xlabel('X Position')
axs[0, 1].set_ylabel('Y Position')
axs[0, 1].legend()
# axs[0, 1].set_title('Filtered Trajectory')

# Plot the estimated trajectory from the smoother
axs[1, 0].scatter(SMOOTHER_X_AXIS, SMOOTHER_Y_AXIS, label='Smoothed Trajectory', color='#3a86ff', marker='s', s=10)
axs[1, 0].set_xlabel('X Position')
axs[1, 0].set_ylabel('Y Position')
axs[1, 0].legend()
# axs[1, 0].set_title('Smoothed Trajectory')

# Plot all trajectories together
axs[1, 1].scatter(X_AXIS, Y_AXIS, label='True Trajectory', color='#ffbe0b', marker='o', s=5)
axs[1, 1].scatter(FILTER_X_AXIS, FILTER_Y_AXIS, label='Filtered Trajectory', color='#ff006e', marker='x', s=5)
axs[1, 1].scatter(SMOOTHER_X_AXIS, SMOOTHER_Y_AXIS, label='Smoothed Trajectory', color='#3a86ff', marker='s', s=5)
axs[1, 1].set_xlabel('X Position')
axs[1, 1].set_ylabel('Y Position')
axs[1, 1].legend()
# axs[1, 1].set_title('All Trajectories')

# Show the plot
plt.tight_layout()
CAR_MODEL_FIG_PATH = "final_result/dynamic_model_of_car/figs/estimated_car_trajectory.pdf"
os.makedirs(os.path.dirname(CAR_MODEL_FIG_PATH), exist_ok=True)
plt.savefig(CAR_MODEL_FIG_PATH)

logger.info(f"Saved the plot to {CAR_MODEL_FIG_PATH}")

# 5. Fit the estimation model to the data with missing values
NEG_LOG_LIKELIHOOD = {
    "True": -CAR_MODEL.loglikelihood(Y=Y, **MODEL_PARAMS),
    "EM": None,
    "GraphEM Laplace": None,
    "GraphEM Gaussian": None,
    "GraphEM Laplace+Gaussian": None,
}

FNORM = {
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
missing_vars = ["A", "H", "Q", "R"]
results = ALG_EM.parameter_estimation(missing_vars=missing_vars, Y=Y, num_iteration=NUM_ITERATION)

NEG_LOG_LIKELIHOOD["EM"] = results["A NegLoglikelihood"][1:]
TRANSITION_MATRIX["EM"] = results["A"][-1]
FNORM['EM'] = results["A Fnorm"][1:]

# REG_LIST = ["Laplace", "Gaussian", "Laplace+Gaussian"]
# ALG_GRAPHEM = GraphEMforA(**MODEL_PARAMS)
# for reg in REG_LIST:

#     graphEM_config = {
#         "reg_type": reg,
#         "num_iteration": NUM_ITERATION,
#         "gamma": 1e-3,  # Douglas-Rachford control parameter
#         "lambda": 50,  # penalty/prior control parameter
#         "eps": 1e-5,
#         "xi": 1e-5,
#     }
#     results = ALG_GRAPHEM.parameter_estimation(Y=Y, **graphEM_config)

#     NEG_LOG_LIKELIHOOD[f"GraphEM {reg}"] = results["A NegLoglikelihood"][1:]
#     TRANSITION_MATRIX[f"GraphEM {reg}"] = results["A"][-1]

# 6. Plot the negative log likelihoods for different estimation methods
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# methods = ["EM", "GraphEM Laplace", "GraphEM Gaussian", "GraphEM Laplace+Gaussian"]

# for i, method in enumerate(methods):
#     ax = axs[i // 2, i % 2]
#     ax.plot(NEG_LOG_LIKELIHOOD[method], label=f'{method}')
#     ax.axhline(y=NEG_LOG_LIKELIHOOD["True"], color='r', linestyle='--', label='True')
#     # ax.set_title(f'{method}')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('NegLogLikelihood')
#     ax.legend()

# plt.tight_layout()

# Plot
fig = plt.figure(figsize=(10, 5))

# Plot neg log likelihood + Fnorm

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(NEG_LOG_LIKELIHOOD["EM"], label='EM', color='#ffbe0b')
ax1.axhline(y=NEG_LOG_LIKELIHOOD["True"], color='r', linestyle='--', label='True')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('NegLogLikelihood')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(FNORM["EM"], label='EM', color='#3a86ff')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Frobenius Norm')
ax2.legend()

plt.tight_layout()

NEG_LOG_LIKELIHOOD_FIG_PATH = "final_result/dynamic_model_of_car/figs/neg_log_likelihood.pdf"
os.makedirs(os.path.dirname(NEG_LOG_LIKELIHOOD_FIG_PATH), exist_ok=True)
plt.savefig(NEG_LOG_LIKELIHOOD_FIG_PATH)

logger.info(f"Saved the plot to {NEG_LOG_LIKELIHOOD_FIG_PATH}")

for plot_var in ["A", "H", "Q", "R"]:
    # Plot
    fig = plt.figure(figsize=(10, 5))

    # Plot neg log likelihood + Fnorm

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(results[f'{plot_var} NegLoglikelihood'][1:], label='EM', color='#ffbe0b')
    ax1.axhline(y=NEG_LOG_LIKELIHOOD["True"], color='r', linestyle='--', label='True')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('NegLogLikelihood')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(results[f'{plot_var} Fnorm'][1:], label='EM', color='#3a86ff')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Frobenius Norm')
    ax2.legend()

    plt.tight_layout()

    neg_log_likelihood_plus_fnorm_path = f"final_result/dynamic_model_of_car/figs/multi_vars_est/{plot_var}_neg_log_likelihood_fnorm.pdf"
    os.makedirs(os.path.dirname(neg_log_likelihood_plus_fnorm_path), exist_ok=True)
    plt.savefig(neg_log_likelihood_plus_fnorm_path)

    logger.info(f"Saved the plot to {neg_log_likelihood_plus_fnorm_path}")


# 6. Plot the trajectory of the car, the estimated trajectory, and the estimated trajectory with missing values
def draw_weighted_directed_graph(adj_matrix, seed, position=None, node_size=50, node_color='#FEDA8B', 
                                 edge_color='#F4A582', edge_width_scale=2, arrow_size=5, font_size=6, 
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
fig = plt.figure(figsize=(8, 4))
# fig.suptitle(f"{graph} graphic results with different reg terms", fontsize=16)

plt.subplot(1, 2, 1)
pos = draw_weighted_directed_graph(A, seed=seed, title=None)

plt.subplot(1, 2, 2)
draw_weighted_directed_graph(TRANSITION_MATRIX["EM"], seed=seed, position=pos, title=None)
plt.tight_layout()

GRAPHIC_FIG_PATH = "final_result/dynamic_model_of_car/figs/graphs_for_true_and_EM.pdf"
os.makedirs(os.path.dirname(GRAPHIC_FIG_PATH), exist_ok=True)
plt.savefig(GRAPHIC_FIG_PATH)

logger.info(f"Saved the plot to {GRAPHIC_FIG_PATH}")

print(np.round(TRANSITION_MATRIX["EM"], 3))

for plot_var in ["A", "H", "Q", "R"]:
    # Plot
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    pos = draw_weighted_directed_graph(MODEL_PARAMS[plot_var], seed=seed, title=None)

    plt.subplot(1, 2, 2)
    draw_weighted_directed_graph(results[f'{plot_var}'][-1], seed=seed, position=pos, title=None)
    plt.tight_layout()

    graph_path = f"final_result/dynamic_model_of_car/figs/multi_vars_est/{plot_var}_graphs_for_true_and_EM.pdf"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    plt.savefig(graph_path)

    logger.info(f"Saved the plot to {graph_path}")
