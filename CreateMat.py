"""

To test further transition matrix, we should make other params consistent as much as possible.

1. demo default A
2. 

"""

import numpy as np

dim_x = 16
A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
H = np.eye(dim_x)  # H matrix as identity
R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
m0 = np.zeros(dim_x)  # Zero vector for mu_0
P0 = np.eye(dim_x) * 0.01  # Small values in P_0

import os
os.makedirs("./data/demo", exist_ok=True)

# Save the matrices to files
np.save("./data/demo/A.npy", A)  # Save matrix A
np.save("./data/demo/Q.npy", Q)  # Save matrix Q
np.save("./data/demo/H.npy", H)  # Save matrix H
np.save("./data/demo/R.npy", R)  # Save matrix R
np.save("./data/demo/m0.npy", m0)  # Save vector m0
np.save("./data/demo/P0.npy", P0)  # Save matrix P0

print("Matrices saved successfully!")
