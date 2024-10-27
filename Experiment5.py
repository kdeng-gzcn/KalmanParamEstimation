from Model import KalmanClass
import numpy as np

model = KalmanClass.MAPParameterEstimationA(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))
    
A, As, metric = model.parameter_estimation(alpha=0.01, numerical=False)
print(As)
print(metric)

model = KalmanClass.EMParameterEstimationA(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))

A, As, metric = model.parameter_estimation(num_iteration=10)
print(As)
print("A10: ", A)
print(metric)