# This is to test High Dimension Parameter Estimation
# try 2*2 A and corresponding parameters

import numpy as np
from Model import KalmanClass
import pandas as pd

# Define the experiment function
def run_experiment(method, A_init, Sigma_q, H, Sigma_r, mu_0, P_0, num_iteration=20, alpha=None):
    if method == "Gradient":
        model = KalmanClass.GradientParameterEstimationAll(var="A", A=A_init, Sigma_q=Sigma_q, H=H, Sigma_r=Sigma_r, mu_0=mu_0, P_0=P_0)
        A, As, metrics = model.parameter_estimation(alpha=alpha, num_iteration=num_iteration)
    elif method == "EM":
        model = KalmanClass.EMParameterEstimationAll(var="A", A=A_init, Sigma_q=Sigma_q, H=H, Sigma_r=Sigma_r, mu_0=mu_0, P_0=P_0)
        A, As, metrics = model.parameter_estimation(num_iteration=num_iteration)

    return As, metrics

# Define matrices for different sizes
matrix_sizes = [2, 3, 4]  # Sizes for 2x2, 3x3, 4x4
num_iteration = 20
alpha = 0.01
num_runs = 10  # Number of repeated runs to average results

# Initialize a list to collect averaged results
averaged_results_list = []

for size in matrix_sizes:
    # Set parameters based on the current matrix size
    A_init = np.eye(size) * 0.9  # Initial A matrix, scaled identity matrix
    Sigma_q = np.eye(size) * 0.01  # Small noise in Sigma_q
    H = np.eye(size)  # H matrix as identity
    Sigma_r = np.eye(size) * 0.01  # Small noise in Sigma_r
    mu_0 = np.zeros(size)  # Zero vector for mu_0
    P_0 = np.eye(size) * 0.01  # Small values in P_0

    # Initialize lists to accumulate metrics and matrices over multiple runs
    gradient_metrics_all = []
    gradient_As_all = []
    em_metrics_all = []
    em_As_all = []

    # Run the experiment multiple times and accumulate results
    for _ in range(num_runs):
        # Gradient method
        As, metrics = run_experiment("Gradient", A_init, Sigma_q, H, Sigma_r, mu_0, P_0, num_iteration, alpha)
        gradient_metrics_all.append(metrics)
        gradient_As_all.append(As)

        # EM method
        As, metrics = run_experiment("EM", A_init, Sigma_q, H, Sigma_r, mu_0, P_0, num_iteration)
        em_metrics_all.append(metrics)
        em_As_all.append(As)

    # Calculate average metrics and matrices over all runs
    gradient_metrics_avg = np.mean(gradient_metrics_all, axis=0)
    em_metrics_avg = np.mean(em_metrics_all, axis=0)
    
    # Record average results for key iterations (e.g., 0, 5, 10, 20)
    key_iterations = [1, 5, 10, 20]
    for i in key_iterations:
        if i < num_iteration+1:
            averaged_results_list.append({
                "Method": "Gradient",
                "Size": f"{size}x{size}",
                "Iteration": i,
                "A": np.mean([A[i] for A in gradient_As_all], axis=0),  # Averaged A matrix at iteration i
                "Metric": gradient_metrics_avg[i]  # Averaged Metric at iteration i
            })
            averaged_results_list.append({
                "Method": "EM",
                "Size": f"{size}x{size}",
                "Iteration": i,
                "A": np.mean([A[i] for A in em_As_all], axis=0),  # Averaged A matrix at iteration i
                "Metric": em_metrics_avg[i]  # Averaged Metric at iteration i
            })

# Convert the results list to a DataFrame
averaged_results = pd.DataFrame(averaged_results_list)

# Save the averaged results to a CSV file
path_csv = "./Result/Experiment5_averaged.csv"
averaged_results.sort_values(["Method", "Size", "Iteration"]).to_csv(path_or_buf=path_csv)

# Print averaged results
print(averaged_results.sort_values(["Method", "Size", "Iteration"]))
