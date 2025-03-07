import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.KalmanProcess import KalmanProcess
from src.EM import EMParameterEstimation
from src.GraphEM import GraphEMforA
import src.GraphEM.funcs_GraphEM as F
from src.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

seed = np.random.seed(42)

def _draw_weighted_directed_graph(adj_matrix, seed, position=None, node_size=50, node_color='#FEDA8B', 
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

DATA_PATH = Path("final_result/graph_types/data")

GRAPH_TYPE = ["Small World", "Scale Free", "Bipartite", "Cycle", "Star"]

for graph_type in GRAPH_TYPE:

    logger.info(f"Start with {graph_type}")

    MODEL_PARAMS = {
        "A": np.load(DATA_PATH / graph_type / "A.npy"),
        "Q": np.load(DATA_PATH / graph_type / "Q.npy"),
        "H": np.load(DATA_PATH / graph_type / "H.npy"),
        "R": np.load(DATA_PATH / graph_type / "R.npy"),
        "m0": np.load(DATA_PATH / graph_type / "m0.npy"),
        "P0": np.load(DATA_PATH / graph_type / "P0.npy"),
    }

    logger.info(f"The shape of {MODEL_PARAMS['A'].shape}")
    logger.info(f"The shape of {MODEL_PARAMS['Q'].shape}")
    logger.info(f"The shape of {MODEL_PARAMS['H'].shape}")
    logger.info(f"The shape of {MODEL_PARAMS['R'].shape}")
    logger.info(f"The shape of {MODEL_PARAMS['m0'].shape}")
    logger.info(f"The shape of {MODEL_PARAMS['P0'].shape}")

    # 2. Generate data from the model
    CAR_MODEL = KalmanProcess(**MODEL_PARAMS)
    DATA = CAR_MODEL.generate_measurement(T=100)

    Y = CAR_MODEL.Y
    # Y = data["Y 1:T"]

    # 3. Fit the model to the data
    FILTER = CAR_MODEL.Filter(Y=Y)
    SMOOTHER = CAR_MODEL.Smoother(Y=Y)

    # 4. Plot the trajectory of the car and the estimated trajectory
    # 5. Fit the estimation model to the data with missing values
    NEG_LOG_LIKELIHOOD = {
    "EM True": -CAR_MODEL.loglikelihood(Y=Y, **MODEL_PARAMS),
    "EM": None,
    "GraphEM Laplace": None,
    "GraphEM Laplace True": None,
    "GraphEM Gaussian": None,
    "GraphEM Gaussian True": None,
    "GraphEM Laplace+Gaussian": None,
    "GraphEM Laplace+Gaussian True": None,
    }
    
    REG_TERM = {
    "Laplace": F.L1_wrt_A,
    "Gaussian": F.Gaussian_Prior_wrt_A,
    "Block Laplace": F.Block_L1_wrt_A,
    "Laplace+Gaussian": F.L1_plus_Gaussian_Prior_wrt_A,
    }

    FNORM = {
        "EM": None,
        "GraphEM Laplace": None,
        "GraphEM Gaussian": None,
        "GraphEM Laplace+Gaussian": None,
    }

    TRANSITION_MATRIX = {
        "True": MODEL_PARAMS["A"],
        "EM": None,
        "GraphEM Laplace": None,
        "GraphEM Gaussian": None,
        "GraphEM Laplace+Gaussian": None,
    }

    NUM_ITERATION = 30

    ALG_EM = EMParameterEstimation(**MODEL_PARAMS)
    missing_vars = ["A"]
    results = ALG_EM.parameter_estimation(missing_vars=missing_vars, Y=Y, num_iteration=NUM_ITERATION)

    NEG_LOG_LIKELIHOOD["EM"] = results["A NegLoglikelihood"][1:]
    TRANSITION_MATRIX["EM"] = results["A"][-1]
    FNORM["EM"] = results["A Fnorm"][-1] # only final value

    REG_LIST = ["Laplace", "Gaussian", "Laplace+Gaussian"]
    for reg in REG_LIST:
        logger.info(f"In graph {graph_type}, the Lt terom with {reg} is {REG_TERM[reg](A=MODEL_PARAMS['A'])}")

    ALG_GRAPHEM = GraphEMforA(**MODEL_PARAMS)
    for reg in REG_LIST:

        graphEM_config = {
            "reg_type": reg,
            "num_iteration": NUM_ITERATION,
            "gamma": 1,  # Douglas-Rachford control parameter
            "lambda": 0.8,  # penalty/prior control parameter 0.9
            "eps": 1e-3,
            "xi": 1e-3,
        }
        results = ALG_GRAPHEM.parameter_estimation(Y=Y, **graphEM_config)

        NEG_LOG_LIKELIHOOD[f"GraphEM {reg} True"] = -CAR_MODEL.loglikelihood(Y=Y, **MODEL_PARAMS) + REG_TERM[reg](A=MODEL_PARAMS["A"])
        NEG_LOG_LIKELIHOOD[f"GraphEM {reg}"] = results["A NegLoglikelihood"][1:]
        TRANSITION_MATRIX[f"GraphEM {reg}"] = results["A"][-1]
        FNORM[f"GraphEM {reg}"] = results["A Fnorm"][-1]

    figsize = (8, 4)

    # 6. Plot the negative log likelihoods for different estimation methods
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    methods = ["EM", "GraphEM Laplace", "GraphEM Gaussian", "GraphEM Laplace+Gaussian"]

    for i, method in enumerate(methods):
        ax = axs[i // 2, i % 2]
        ax.plot(NEG_LOG_LIKELIHOOD[method], label=f'{method}', color="#ffb703")
        ax.axhline(y=NEG_LOG_LIKELIHOOD[f"{method} True"], color='r', linestyle='--', label='$\mathcal{L}_T(\mathbf{A})$')
        ax.set_xlabel('Iteration $n$')
        ax.set_ylabel('$\mathcal{L}_T(\mathbf{A}^{(n)})$')
        ax.legend()

    plt.tight_layout()
    NEG_LOG_LIKELIHOOD_FIG_PATH = f"final_result/graph_types/figs/{graph_type}/neg_log_likelihood.pdf"
    os.makedirs(os.path.dirname(NEG_LOG_LIKELIHOOD_FIG_PATH), exist_ok=True)
    plt.savefig(NEG_LOG_LIKELIHOOD_FIG_PATH)

    logger.info(f"Saved the plot to {NEG_LOG_LIKELIHOOD_FIG_PATH}")

    # 6. Plot the trajectory of the car, the estimated trajectory, and the estimated trajectory with missing values

    figsize = (6, 8)
    # plt.style.use('default')
    fig = plt.figure(figsize=figsize)
    # fig.suptitle(f"{graph} graphic results with different reg terms", fontsize=16)

    plt.subplot(3, 2, 1)
    pos = _draw_weighted_directed_graph(np.load(DATA_PATH / graph_type / "A_raw.npy"), seed=seed, title="Transition Matrix", edge_color="#be95c4")

    plt.subplot(3, 2, 3)
    _draw_weighted_directed_graph(TRANSITION_MATRIX["EM"], seed=seed, position=pos, title="EM")

    for i, method in enumerate(methods):
        plt.subplot(3, 2, i+3)
        _draw_weighted_directed_graph(TRANSITION_MATRIX[method], seed=seed, position=pos, title=f"{method}")

    plt.tight_layout()

    GRAPHIC_FIG_PATH = f"final_result/graph_types/figs/{graph_type}/graphs_for_true_and_EM.pdf"
    os.makedirs(os.path.dirname(GRAPHIC_FIG_PATH), exist_ok=True)
    plt.savefig(GRAPHIC_FIG_PATH)

    logger.info(f"Saved the plot to {GRAPHIC_FIG_PATH}")

    # print(np.round(TRANSITION_MATRIX["EM"], 3))

    method_list = ["EM", "GraphEM Laplace", "GraphEM Gaussian", "GraphEM Laplace+Gaussian"]

    data = {
        r"\textbf{Graph Type}": [graph_type] + [None]* (len(method_list) - 1),
        r"\textbf{Method}": method_list,
        r"\textbf{$\mathcal{L}_T(\widehat{\mathbf{A}})$}":[NEG_LOG_LIKELIHOOD[method][-1] for method in method_list],
        r"\textbf{$\| \widehat{\mathbf{A}} - \mathbf{A} \|_F$}": [FNORM[method] for method in method_list],
    }

    df = pd.DataFrame(data)

    # df.index = ["Block Diagonal"] * len(method_list)
    # df.index.name = r"\textbf{Graph Type}"

    TABLE_PATH = f"final_result/graph_types/table/{graph_type}/result_table.tex"
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    df.to_latex(
        TABLE_PATH,
        index=False,
        na_rep="",
        float_format="%.3f",
        column_format="l" * len(data),
        position="tb",
        caption="Result for blockwise-diagonal graph with different regularizations.",
        label="tab: prior results for block-diag",
    )

    logger.info(f"table saved to {TABLE_PATH}")
