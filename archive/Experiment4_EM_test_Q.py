"""

EM Alg for testing high dim Q

"""

from Model import KalmanClass
from matplotlib import pyplot as plt

import numpy as np
np.random.seed(42)

num_iteration = 20

dim_x = 2
A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
H = np.eye(dim_x)  # H matrix as identity
R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
m0 = np.zeros(dim_x)  # Zero vector for mu_0
P0 = np.eye(dim_x) * 0.01  # Small values in P_0

model = KalmanClass.EMParameterEstimationAll(var="Q", A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)

_, Q_list, fnorm_list, neg_loglikelihood_list = model.parameter_estimation(num_iteration=num_iteration)

# Set style
plt.style.use("ggplot")
fig = plt.figure(figsize=(16, 12))  # Create figure with adjusted height for four subplots
fig.suptitle(rf"EM Algorithm for Missing Variable, num_iteration = {num_iteration}")

# Dynamically add the first ax (for A)
ax1 = fig.add_subplot(1, 1, 1)  # Position of the first ax in a 2x2 grid

# Plot loglikelihoods for Q

# Add a vertical line at the final estimated value of Q
for Q in Q_list:
    print(Q)
    # # Add a vertical line at the final estimated value of Q
    # ax1.axvline(x=Q, 
    #             linestyle="--",
    #             label=rf"$\hat{{Q}}^{{({num_iteration})}}: {Q_list[-1].squeeze(): .3f}$")

# Set title and labels for the third ax
ax1.set_title(rf"Missing Variable: $Q$")
ax1.set_xlabel(r"$Q$")
ax1.set_ylabel(r"$ \ell ( Q \mid Y, Q ) $")
ax1.legend()
