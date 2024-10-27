# 1. to plot likelihood (actually pdf for y given A) wrt A
# 2. try MLE estimation in closed-form
# 3. try EM algorithm 

from Model.KalmanClass import ParameterEstimation
import numpy as np
from matplotlib import pyplot as plt
# import autograd.numpy as np
# from autograd import grad
# from scipy.optimize import approx_fprime

def load_model(A = np.array([[1.]]), H = np.array([[1.]]), Sigma_q = np.diag([0.01]), Sigma_r = np.diag([0.01]), mu = np.array([0]), Sigma_p = np.diag([0.01])):

    Model = ParameterEstimation(
        A=A,
        Sigma_q=Sigma_q,
        H=H,
        Sigma_r=Sigma_r,
        mu = mu,
        Sigma_p = Sigma_p
    )

    return Model

def ell(A, Y):

    model = load_model(A=A)
    ell = model.loglikelihood(Y=Y)

    return np.array(ell)

def numerical_gradient_ell(A, Y, epsilon=1e-6):
    
    f_A_plus_eps = ell(A + epsilon, Y)
    f_A_minus_eps = ell(A - epsilon, Y)
    
    grad_A = (f_A_plus_eps - f_A_minus_eps) / (2 * epsilon)

    return grad_A

def loglikelihoods(As, Y):
    '''
    args:
        As: (num_A, Ashape1, Ashap2)
        Y: (timesteps, dim_y) Y under A is true value
    return:
        ls: (num_A, ) the l for each A and fixed Y
    '''

    loglikelihoods = []

    for A in As:

        l = ell(A, Y)

        loglikelihoods.append(l)

    return loglikelihoods

def iteration(alpha, Y):

    A = np.random.randn(1, 1)

    As = [A]

    for _ in range(10):

        A = A + alpha * numerical_gradient_ell(A, Y)

        As.append(A)

    return A, As

def experiment(alpha):

    model = load_model()

    samples = model.generate_measurement(
        mu=model.kf.initial_state_mean,
        Sigma_p=model.kf.initial_state_covariance
    )

    Y = samples['Ytm']
    Y = np.array(Y)

    As_for_plot = [np.array([[A]]) for A in np.linspace(0, 2, 100)]

    ls = loglikelihoods(As_for_plot, Y)

    A, As = iteration(alpha=alpha, Y=Y)

    plt.style.use('ggplot')  # Apply ggplot style

    plt.figure(figsize=(10, 6))
    
    plt.plot(np.array(As_for_plot).squeeze(), ls, label=r"$\ell(A \mid Y)$")
    plt.axvline(x=model.A.squeeze(), color='b', linestyle='--', label='True Value')
    # print(As, len(As))
    for idx, A in enumerate(As):
        plt.axvline(x=A.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {A.squeeze()}')

    plt.title("Loglikelihood wrt A")
    plt.xlabel(r"$A$")
    plt.ylabel(r"$\ell(A \mid Y)$")
    plt.legend()

    # plt.savefig('ell iteration demo.png', dpi=300, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":

    experiment(alpha=0.001)   
