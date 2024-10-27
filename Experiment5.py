# This is to test High Dimension Parameter Estimation
# try 2*2 A and corresponding parameters

from Model import KalmanClass
import numpy as np

path = "./Result/Experiment5.txt"

with open(path, "a") as f:

    model = KalmanClass.MAPParameterEstimationA(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))
        
    A, As, metric = model.parameter_estimation(alpha=0.01, numerical=False, num_iteration=10)
    print(As)
    print(metric)

    f.write("Gradient Method\n")
    f.write(f"A0: {As[0]}, \nA1: {As[1]}, \nA2: {As[2]}, \nA5: {As[5]}, \nA10: {As[10]} \n")
    f.write(f"metric0: {metric[0]}, \nmetric1: {metric[1]}, \nmetric2: {metric[2]}, \nmetric5: {metric[5]}, \nmetric10: {metric[10]} \n")

    model = KalmanClass.EMParameterEstimationA(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))

    A, As, metric = model.parameter_estimation(num_iteration=10)
    print(As)
    print("A10: ", A)
    print(metric)

    f.write("EM Method\n")
    f.write(f"A0: {As[0]}, \nA1: {As[1]}, \nA2: {As[2]}, \nA5: {As[5]}, \nA10: {As[10]} \n")
    f.write(f"metric0: {metric[0]}, \nmetric1: {metric[1]}, \nmetric2: {metric[2]}, \nmetric5: {metric[5]}, \nmetric10: {metric[10]} \n")