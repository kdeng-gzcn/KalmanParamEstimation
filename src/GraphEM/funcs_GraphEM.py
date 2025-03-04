import numpy as np
import scipy.linalg as la

"""

Q(A, An) = -T / 2 * trace(Q^(-1) * (Sigma - C @ A.T - A @ C.T + A @ Phi @ A.T))

"""

def q_wrt_A_given_An(Q=None, A=None, Sigma=None, Phi=None, C=None, T=None):

    # sigma_Q = np.diag(Q)[0]
    Q_inv = np.linalg.inv(Q)

    # q = K / 2 * np.trace((1 / sigma_Q**2) * (Sigma - C @ D1.T - D1 @ C.T + D1 @ Phi @ D1.T))
    q = -T / 2 * np.trace(Q_inv @ (Sigma - C @ A.T - A @ C.T + A @ Phi @ A.T))
    
    return q

"""

Prox Q at Ai == argmin_A Q(A, Ai)

"""

def prox_gamma_minus_Q_wrt_Ai(**kwargs):
    """
    again, set gamma as 1

    this time we dont need gamma from norm in obj function, but we can solve it with explicit formula

    prox q at A
    """

    A = kwargs.get("A", None)
    C = kwargs.get("C", None)
    Phi = kwargs.get("Phi", None)
    Q_Cov = kwargs.get("Q", None)
    gamma = kwargs.get("gamma", None)
    T = kwargs.get("T", None)

    sigma_Q_sq = np.diag(Q_Cov)[0]
    coef = T * gamma / sigma_Q_sq

    argmin_A = (coef * C + A) @ np.linalg.inv(coef * Phi + np.eye(Phi.shape[0]))

    # X = gamma * np.linalg.inv(Q_Cov)
    # Y = np.linalg.inv(Phi)
    # Z = A @ np.linalg.inv(Phi) + gamma * np.linalg.inv(Q_Cov) @ C @ np.linalg.inv(Phi)

    # argmin_A = la.solve_continuous_lyapunov(X, Z - Y @ X)  # X * A + A * Y = Z

    return argmin_A

"""

Regular Term

"""

def L1_wrt_A(A=None):
    Reg1 = np.sum(np.abs(A))  # L1 norm case

    return Reg1

def Gaussian_Prior_wrt_A(A=None):
    """
    gaussian prior is 1/2 || ||F
    """
    Reg1 = 1/2 * np.linalg.norm(A, 'fro')  # L1 norm + Gaussian case

    return Reg1

def Block_L1_wrt_A(A=None):
    pass

def L1_plus_Gaussian_Prior_wrt_A(A=None):
    reg_value = (np.sum(np.abs(A)) + 1/2 * np.linalg.norm(A, 'fro'))

    return reg_value

"""

Prox Regular Term at A == argmin_A L

"""

def prox_gamma_L1_wrt_Ai(gamma=None, A=None):
    '''
    set gamma for norm inside optim method as 1

    note that arg gamma is for norm in obj function

    prox L1 at A
    '''
    Aprox = np.sign(A) * np.maximum(0, np.abs(A) - gamma)

    return Aprox

def prox_gamma_L2_wrt_Ai(gamma=None, A=None):

    argmin = A / (1 + gamma)

    return argmin

def opt_wrt_Block_L1(A=None, gamma=None):

    pass

def prox_gamma_L1_plus_L2_wrt_Ai(gamma=None, A=None):

    '''

    set gamma for norm inside optim method as 1

    note that arg gamma is for norm in obj function
    
    '''
    
    argmin = np.sign(A / (1 + gamma)) * np.maximum(0, np.abs(A / (1 + gamma)) - (gamma / (1 + gamma)))

    return argmin


"""

High level function for object GraphEM Q

"""

def Q_wrt_A_given_An(**kwargs):

    REG_TERM = {
        "Laplace": L1_wrt_A,
        "Gaussian": Gaussian_Prior_wrt_A,
        "Block Laplace": Block_L1_wrt_A,
        "Laplace+Gaussian": L1_plus_Gaussian_Prior_wrt_A,
    }

    REG_TYPE = kwargs.get("reg_type", "Laplace")
    A = kwargs.get("A", None)
    Q_COV = kwargs.get("Q", None)
    SIGMA = kwargs.get("Sigma", None)
    PHI = kwargs.get("Phi", None)
    C = kwargs.get("C", None)
    T = kwargs.get("T", None)
    LAMBDA = kwargs.get("lambda", None)

    q = q_wrt_A_given_An(Q=Q_COV, A=A, Sigma=SIGMA, Phi=PHI, C=C, T=T) 
    reg_term = REG_TERM[REG_TYPE](A=A) 

    Q_OBJ = - q + LAMBDA * (reg_term)

    return Q_OBJ


if __name__ == "__main__":
    
    pass
