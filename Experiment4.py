# This experiment is to test the other parameter's estimation
# H works
# Q, R and P_0 are positive, which is hard for gradient method
# m_0 needs long recursion, which is hard to calculate the gradient

from Model import KalmanClass
from utils import Plotter
from matplotlib import pyplot as plt
import numpy as np

# ### Gradient

# model = KalmanClass.GradientParametersEstimationAll(vars="H")

# print(model.mu_0, model.theta)

# Theta, Thetas = model.parameter_estimation(alpha=0.001, num_iteration=10)
# print(model.theta, Thetas)

# Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 2))

# plt.style.use("ggplot")
# plt.figure(figsize=(10, 6))

# plt.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
# # plt.axvline(x=model.A.squeeze(), color='b', linestyle='--', label=f'True Value {model.A.squeeze()}')
# for idx, t in enumerate(Thetas):
#     plt.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

# plt.title("Gradient Parameter Estimation for $H$")
# plt.xlabel(r"$\theta$")
# plt.ylabel(r"$ \ell ( \theta \mid Y) $")
# plt.legend()

# plt.savefig("./Result/Gradient Parameter Estimation for H demo.png")

# plt.show()

### EM

model = KalmanClass.EMParameterEstimationAll(vars="mu")

Theta, Thetas, metric = model.parameter_estimation(num_iteration=10)
print("missing value:", model.theta + ",", "Thetas:", Thetas)

Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 2))

plt.style.use("ggplot")
fig = plt.figure(figsize=(12, 16))
fig.suptitle("EM Parameters Estimation", fontsize=16)

for i in range(1, 7):
    ax = fig.add_subplot(3, 2, i)
    if i == 1:
        model = KalmanClass.EMParameterEstimationAll(vars="A")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 2))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.A.squeeze(), color='b', linestyle='--', label=f'True Value {model.A.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $A$")
    elif i == 2:
        model = KalmanClass.EMParameterEstimationAll(vars="H")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 2))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.H.squeeze(), color='b', linestyle='--', label=f'True Value {model.H.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $H$")
    elif i == 3:
        model = KalmanClass.EMParameterEstimationAll(vars="Q")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 0.02))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.Sigma_q.squeeze(), color='b', linestyle='--', label=f'True Value {model.Sigma_q.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $Q$")
    elif i == 4:
        model = KalmanClass.EMParameterEstimationAll(vars="R")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 0.02))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.Sigma_r.squeeze(), color='b', linestyle='--', label=f'True Value {model.Sigma_r.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $R$")
    elif i == 5:
        model = KalmanClass.EMParameterEstimationAll(vars="mu")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(-1, 1))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.mu_0.squeeze(), color='b', linestyle='--', label=f'True Value {model.mu_0.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $m_0$")
    elif i == 6:
        model = KalmanClass.EMParameterEstimationAll(vars="P")

        Theta, Thetas, metric = model.parameter_estimation(num_iteration=5)
        print("missing value:", model.theta + ",", "Thetas:", Thetas)

        Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(0, 0.02))

        ax.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
        ax.axvline(x=model.P_0.squeeze(), color='b', linestyle='--', label=f'True Value {model.P_0.squeeze()}')
        for idx, t in enumerate(Thetas):
            ax.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$ \ell ( \theta \mid Y) $")
        ax.set_title(r"Missing Value: $P_0$")

plt.tight_layout()
plt.savefig("./Result/EM Parameter Estimation for Other Params.pdf")
# plt.show()
