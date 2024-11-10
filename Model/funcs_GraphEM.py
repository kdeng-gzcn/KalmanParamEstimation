"""
This module is for functions in GraphEM
"""
import numpy as np

def Q():
    pass

def loglikelihood_multi_normal():
    pass

def q_wrt_A(Q=None, A=None, Sigma=None, Phi=None, C=None, T=None):
    """
    q is actually -q
    we use this to get minimum, hence it's -q
    """
    K = T
    sigma_Q = np.diag(Q)[0]
    D1 = A
    q = K / 2 * np.trace((1 / sigma_Q**2) * (Sigma - np.dot(C, D1.T) - np.dot(D1, C.T) + np.dot(np.dot(D1, Phi), D1.T)))
    return q

def L1_wrt_A(A=None, gamma=None):
    """
    gamma is from norm in obj
    """
    Reg1 = gamma * np.sum(np.abs(A))  # L1 norm case
    return Reg1

def opt_wrt_L1(gamma=None, A=None):
    '''
    set gamma for norm inside optim method as 1

    note that arg gamma is for norm in obj function
    '''
    temp = 1 * gamma
    Aprox = np.sign(A) * np.maximum(0, np.abs(A) - temp)
    return Aprox

def opt_wrt_q(A=None, C=None, Phi=None, Q=None, T=None):
    """
    again, set gamma as 1

    this time we dont need gamma from norm in obj function, but we can solve it with explicit formula
    """
    K = T
    sigma_Q = np.diag(Q)[0]
    temp = 1 * K / (sigma_Q ** 2)
    Aprox = np.dot((temp * C + A), np.linalg.inv(Phi * temp + np.eye(Phi.shape[0])))
    return Aprox

