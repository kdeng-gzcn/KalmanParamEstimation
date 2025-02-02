"""

test for experiment9, why ex. 9 failed???

"""

def init_A():

    """
    
    create A with spectral limit
    
    """

    A = None

    return A

# 0. import pkg
import numpy as np
import os
np.random.seed(0)
from matplotlib import pyplot as plt
from Model.GraphEM import GraphEMforA
from Model.KalmanClass import EMParameterEstimationAll

# 1. load model
# 1.1 setting model params and hyper params
dim_x =2
A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
A[1, 0] = 0.1

print(f"modified A: {A}")

Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
H = np.eye(dim_x)  # H matrix as identity
R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
m0 = np.zeros(dim_x)  # Zero vector for mu_0
P0 = np.eye(dim_x) * 0.01  # Small values in P_0

# 1.2 load model
model_MLE = EMParameterEstimationAll(var="A", A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0)
model_list = []
for reg in ["Laplace", "Gaussian", "Laplace_Gaussian"]:
    model = GraphEMforA(A=A, Sigma_q=Q, H=H, Sigma_r=R, mu_0=m0, P_0=P0, reg_name=reg)
    model.Y = model_MLE.Y
    model_list.append(model)
# 1.3 run model and get results
"""
return {"A iterations": A_list, "Fnorm iterations": Fnorm_list, "Simple Q iterations": obj_list, "General Q iteratioins": None, "Loglikelihood iterations": None}
"""
_, A_list_MLE, Fnorm_list_MLE, Neg_Loglikelihood_list_MLE = model_MLE.parameter_estimation(num_iteration=10)
res_list = []
for idx, reg in enumerate(["Laplace", "Gaussian", "Laplace_Gaussian"]):
    results = model_list[idx].parameter_estimation(num_iteration=10, gamma=0.1, eps=1e-5, xi=1e-5)
    res_list.append(results)

# 2. analysis
# 2.1 unpack results
A_seq_list = []
Fnorm_seq_list = []
Q_seq_list = []
Neg_Loglikelihood_seq_list = []
for idx, reg in enumerate(["Laplace", "Gaussian", "Laplace_Gaussian"]):
    
    A_list, Fnorm_list, Q_list, _, Neg_Loglikelihood_list = res_list[idx].values()

    A_seq_list.append(A_list)
    Fnorm_seq_list.append(Fnorm_list)
    Q_seq_list.append(Q_list)
    Neg_Loglikelihood_seq_list.append(Neg_Loglikelihood_list)

# 2.2 visulization
# 2.2.1 plot fnorm / obj (Q = q + reg) / loglikelihood
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"Tests for GraphEM with different reg term")

baseline = -model.loglikelihood(theta=A, Y=model.Y)

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(Neg_Loglikelihood_list_MLE, c=colors[0], label="MLEM")
ax1.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax1.set_xlabel("t")
ax1.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax1.legend()

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(Neg_Loglikelihood_seq_list[0], c=colors[2], label="GraphEM with Laplace reg")
ax2.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax2.set_xlabel("t")
ax2.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(Neg_Loglikelihood_seq_list[1], c=colors[3], label="GraphEM with Gaussian reg")
ax3.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax3.set_xlabel("t")
ax3.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(Neg_Loglikelihood_seq_list[2], c=colors[4], label="GraphEM with Gaussian + Laplace reg")
ax4.axhline(y=baseline, color=colors[1], linestyle='--', label=r'$ -\ell (A^{ture} \mid Y) $')
ax4.set_xlabel("t")
ax4.set_ylabel(r"$ -\ell (A^{(t)} \mid Y) $")
ax4.legend()

plt.tight_layout()
import os
os.makedirs("./Result/Experiment10/", exist_ok=True)
plt.savefig(f"./Result/Experiment10/Tests with GraphEM With 3 Different Reg Term.pdf")

# 2.2.2 text result
print("True A:")
print(A)

print("Ahat from EM:")
print(A_list_MLE[-1])

print("Baseline:")
print(baseline)

print("Final Neg Loglikelihood MLE:")
print(Neg_Loglikelihood_list_MLE[-1])

print("Final Neg Loglikelihood Laplace Reg:")
print(Neg_Loglikelihood_seq_list[0][-1])

print("Final Neg Loglikelihood Gaussian Reg:")
print(Neg_Loglikelihood_seq_list[1][-1])

print("Final Neg Loglikelihood Laplace+Gaussian Reg:")
print(Neg_Loglikelihood_seq_list[2][-1])

plt.show()
