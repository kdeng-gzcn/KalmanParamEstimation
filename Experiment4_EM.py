# This experiment is to test the other parameter's estimation
# H works
# Q, R and P_0 are positive, which is hard for gradient method
# m_0 needs long recursion, which is hard to calculate the gradient

from Model import KalmanClass
from utils import Plotter
from matplotlib import pyplot as plt

import numpy as np
np.random.seed(42)

num_iteration = 20

# Set style
plt.style.use("ggplot")
fig = plt.figure(figsize=(16, 12))  # Create figure with adjusted height for four subplots
fig.suptitle(rf"EM Algorithm for Missing Variable, num_iteration = {num_iteration}")

# Dynamically add the first ax (for A)
ax1 = fig.add_subplot(2, 2, 1)  # Position of the first ax in a 2x2 grid

for A in [0.9, 0.8, 0.5, 0.3]:
    model = KalmanClass.EMParameterEstimationAll(var="A", A=np.array([[A]]))
    A_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="A", xlim=(-0.5, 1.5))

    A_fianl, As, _ = model.parameter_estimation(num_iteration=num_iteration)

    # Plot loglikelihoods for A
    line, = ax1.plot(
        np.array(A_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( A \mid Y, A = {A} )$"
    )
    
    # Add a vertical line at the final estimated value of A
    ax1.axvline(x=A_fianl, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{A}}^{{({num_iteration})}}: {np.round(A_fianl, 3).squeeze()}$")

# Set title and labels for the first ax
ax1.set_title(rf"Missing Variable: $A$")
ax1.set_xlabel(r"$A$")
ax1.set_ylabel(r"$ \ell ( A \mid Y, A ) $")
ax1.legend()

# Dynamically add the second ax (for H)
ax2 = fig.add_subplot(2, 2, 2)  # Position of the second ax in a 2x2 grid

for H in [1.0, 0.8, 0.5, 0.3]:
    model = KalmanClass.EMParameterEstimationAll(var="H", H=np.array([[H]]))
    H_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="H", xlim=(-0.5, 1.5))

    H_fianl, Hs, _ = model.parameter_estimation(num_iteration=num_iteration)

    # Plot loglikelihoods for H
    line, = ax2.plot(
        np.array(H_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( H \mid Y, H = {H} )$"
    )
    
    # Add a vertical line at the final estimated value of H
    ax2.axvline(x=H_fianl, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{H}}^{{({num_iteration})}}: {np.round(H_fianl, 3).squeeze()}$")

# Set title and labels for the second ax
ax2.set_title(rf"Missing Variable: $H$")
ax2.set_xlabel(r"$H$")
ax2.set_ylabel(r"$ \ell ( H \mid Y, H ) $")
ax2.legend()

# Dynamically add the third ax (for Q)
ax3 = fig.add_subplot(2, 2, 3)  # Position of the third ax in a 2x2 grid

for Q in [0.015, 0.01, 0.005]:
    model = KalmanClass.EMParameterEstimationAll(var="Q", Sigma_q=np.array([[Q]]))
    Q_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="Q", xlim=(0, 0.02))

    Q_final, Qs, _ = model.parameter_estimation(num_iteration=num_iteration)

    # Plot loglikelihoods for Q
    line, = ax3.plot(
        np.array(Q_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( Q \mid Y, Q = {Q} )$"
    )
    
    # Add a vertical line at the final estimated value of Q
    ax3.axvline(x=Q_final, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{Q}}^{{({num_iteration})}}: {np.round(Q_final, 3).squeeze()}$")

# Set title and labels for the third ax
ax3.set_title(rf"Missing Variable: $Q$")
ax3.set_xlabel(r"$Q$")
ax3.set_ylabel(r"$ \ell ( Q \mid Y, Q ) $")
ax3.legend()

# Dynamically add the fourth ax (for R)
ax4 = fig.add_subplot(2, 2, 4)  # Position of the fourth ax in a 2x2 grid

for R in [0.015, 0.01, 0.005]:
    model = KalmanClass.EMParameterEstimationAll(var="R", Sigma_r=np.array([[R]]))
    R_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="R", xlim=(0, 0.02))

    R_final, Rs, _ = model.parameter_estimation(num_iteration=num_iteration)

    # Plot loglikelihoods for R
    line, = ax4.plot(
        np.array(R_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( R \mid Y, R = {R} )$"
    )
    
    # Add a vertical line at the final estimated value of R
    ax4.axvline(x=R_final, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{R}}^{{({num_iteration})}}: {np.round(R_final, 3).squeeze()}$")

# Set title and labels for the fourth ax
ax4.set_title(rf"Missing Variable: $R$")
ax4.set_xlabel(r"$R$")
ax4.set_ylabel(r"$ \ell ( R \mid Y, R ) $")
ax4.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("./Result/EM Algorithm for All Params.pdf")
plt.show()
