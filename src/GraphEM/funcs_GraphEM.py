"""

This module is for functions in GraphEM

"""

import numpy as np

"""

q + reg == approx of loglikelihood (lower bound)

"""

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

# def q_wrt_Q(Q=None, A=None, Sigma=None, Phi=None, C=None, T=None):
#     """
#     q is actually -q
#     we use this to get minimum, hence it's -q
#     """
#     K = T
#     sigma_Q = np.diag(Q)[0]
#     D1 = A
#     q = K / 2 * np.trace((1 / sigma_Q**2) * (Sigma - np.dot(C, D1.T) - np.dot(D1, C.T) + np.dot(np.dot(D1, Phi), D1.T)))
#     return q

"""

Prox q at A == argmin_A q

"""

def opt_wrt_q(A=None, C=None, Phi=None, Q=None, T=None):
    """
    again, set gamma as 1

    this time we dont need gamma from norm in obj function, but we can solve it with explicit formula

    prox q at A
    """
    K = T
    sigma_Q = np.diag(Q)[0]
    temp = 1 * K / (sigma_Q ** 2)
    Aprox = np.dot((temp * C + A), np.linalg.inv(Phi * temp + np.eye(Phi.shape[0])))
    return Aprox

"""

Regular Term

"""

def L1_wrt_A(A=None, gamma=None):
    """
    gamma is from norm in obj
    """
    Reg1 = gamma * np.sum(np.abs(A))  # L1 norm case
    return Reg1

# def L1_wrt_Q(Q=None, gamma=None):
#     """
#     gamma is from norm in obj
#     """
#     Reg1 = gamma * np.sum(np.abs(Q))  # L1 norm case
#     return Reg1

def Gaussian_Prior_wrt_A(A=None, gamma=None):
    """
    gamma is from norm in obj

    gaussian prior is 1/2 || ||F
    """
    Reg1 = gamma * 1/2 * np.linalg.norm(A, 'fro')  # L1 norm + Gaussian case
    return Reg1

def Block_L1_wrt_A(A=None, gamma=None):

    pass

def L1_Gaussian_Prior_wrt_A(A=None, gamma=None):

    reg_value =  gamma * (np.sum(np.abs(A)) + 1/2 * np.linalg.norm(A, 'fro'))

    return reg_value

"""

Prox Regular Term at A == argmin_A L

"""

def opt_wrt_L1(gamma=None, A=None):
    '''
    set gamma for norm inside optim method as 1

    note that arg gamma is for norm in obj function

    prox L1 at A
    '''
    temp = 1 * gamma
    Aprox = np.sign(A) * np.maximum(0, np.abs(A) - temp)
    return Aprox

def opt_wrt_Gaussian_Prior(gamma=None, A=None):

    argmin = A / (1 + gamma)

    return argmin

def opt_wrt_Block_L1(A=None, gamma=None):

    pass

def opt_wrt_L1_Gaussian_Prior(gamma=None, A=None):

    '''

    set gamma for norm inside optim method as 1

    note that arg gamma is for norm in obj function
    
    '''
    
    argmin = np.sign(A / (1 + gamma)) * np.maximum(0, np.abs(A / (1 + gamma)) - (gamma / (1 + gamma)))

    return argmin

# def opt_wrt_L1_given_Q(gamma=None, Q=None):
#     '''
#     set gamma for norm inside optim method as 1

#     note that arg gamma is for norm in obj function
#     '''
#     temp = 1 * gamma
#     Aprox = np.sign(Q) * np.maximum(0, np.abs(Q) - temp)
#     return Aprox

if __name__ == "__main__":
    
    pass
