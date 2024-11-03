# This experiment is to test the other parameter's estimation
# H works
# Q, R and P_0 are positive, which is hard for gradient method
# m_0 needs long recursion, which is hard to calculate the gradient

# Set a random seed for reproducibility
import numpy as np
np.random.seed(42)

# Import other necessary modules
from Model import KalmanClass
from utils import Plotter
from matplotlib import pyplot as plt

### Gradient

# Set style
plt.style.use("ggplot")
fig = plt.figure(figsize=(14, 6))  # Create figure

alpha = 0.01
num_iteration = 20

fig.suptitle(rf"Gradient Algorithm for Missing Variable, $\alpha$={alpha}, num_iteration={num_iteration}")

# Dynamically add the first ax (for H)
ax1 = fig.add_subplot(1, 2, 1)  # [left, bottom, width, height] in proportion

for H in [1.0, 0.6, 0.3]:
    model = KalmanClass.GradientParametersEstimationAll(var="H", H=np.array([[H]]))
    H_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="H", xlim=(-0.5, 1.5))

    H_fianl, Hs = model.parameter_estimation(alpha=alpha, num_iteration=num_iteration, )

    # Plot loglikelihoods for H
    line, = ax1.plot(
        np.array(H_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( H \mid Y, H = {H} )$"
    )

    ax1.axvline(x=Hs[0], 
                linestyle=":",
                color=line.get_color(),
                label=rf"$\hat{{H}}^{{(0)}}: {np.round(Hs[0], 3).squeeze()}$")
    
    # Add a vertical line at the final estimated value of H
    ax1.axvline(x=H_fianl, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{H}}^{{({num_iteration})}}: {np.round(H_fianl, 3).squeeze()}$")

# Set title and labels for the first ax
ax1.set_title(rf"Missing Variable: $H$")
ax1.set_xlabel(r"$H$")
ax1.set_ylabel(r"$ \ell ( H \mid Y, H ) $")
ax1.legend()

# Dynamically add the second ax (for A)
ax2 = fig.add_subplot(1, 2, 2)  # Position of the second ax in proportion

for A in [0.9, 0.6, 0.3]:
    model = KalmanClass.GradientParametersEstimationAll(var="A", A=np.array([[A]]))
    A_for_plots, loglikelihoods = model.data_for_plot_loglikelihood(theta_name="A", xlim=(-0.5, 1.5))

    A_fianl, As = model.parameter_estimation(alpha=alpha, num_iteration=num_iteration)

    # Plot loglikelihoods for A
    line, = ax2.plot(
        np.array(A_for_plots).squeeze(), 
        loglikelihoods, 
        label=fr"$\ell ( A \mid Y, A = {A} )$"
    )

    # Add a vertical line at the final estimated value of A
    ax2.axvline(x=As[0], 
                linestyle=":",
                color=line.get_color(),
                label=rf"$\hat{{A}}^{{(0)}}: {np.round(As[0], 3).squeeze()}$")
    
    # Add a vertical line at the final estimated value of A
    ax2.axvline(x=A_fianl, 
                linestyle="--",
                color=line.get_color(),
                label=rf"$\hat{{A}}^{{({num_iteration})}}: {np.round(A_fianl, 3).squeeze()}$")

# Set title and labels for the second ax
ax2.set_title(rf"Missing Variable: $A$")
ax2.set_xlabel(r"$A$")
ax2.set_ylabel(r"$ \ell ( A \mid Y, A ) $")
ax2.legend()

# Display the figure
plt.tight_layout()
plt.savefig("./Result/Gradient Algorithm for All Params.pdf")
plt.show()
