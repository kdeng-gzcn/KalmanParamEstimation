from Model import KalmanClass
from utils import Plotter
from matplotlib import pyplot as plt
import numpy as np

model = KalmanClass.GradientParametersEstimationAll(vars="mu")

print(model.mu_0, model.theta)

Theta, Thetas = model.parameter_estimation(alpha=0.01, num_iteration=15)
print(model.theta, Thetas)

Thetas_for_plot, loglikelihoods = model.data_for_plot_loglikelihood(xlim=(-1, 1))

plt.style.use("ggplot")
plt.figure(figsize=(10, 6))

plt.plot(np.array(Thetas_for_plot).squeeze(), loglikelihoods, label=r"$\ell(\theta \mid Y)$")
# plt.axvline(x=model.A.squeeze(), color='b', linestyle='--', label=f'True Value {model.A.squeeze()}')
for idx, t in enumerate(Thetas):
    plt.axvline(x=t.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(t.squeeze(), 3)}')

plt.title("H")
plt.xlabel("H")
plt.ylabel("loglikelihood")
plt.legend()

plt.show()
