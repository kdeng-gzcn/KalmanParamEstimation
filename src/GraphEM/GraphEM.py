# for local import
import sys
sys.path.append("")

import logging

import numpy as np

from src.KalmanProcess import KalmanProcess
import src.GraphEM.funcs_GraphEM as F
from src.logging.logging_config import setup_logging

class GraphEMforA(KalmanProcess):

    def __init__(self, **kwargs):

        # setup_logging()
        self.logger = logging.getLogger(__name__)

        # set up true model params and get X, Y
        super().__init__(**kwargs)

    def _Douglas_Rachford(self, **kwargs):

        """

        The core process from MATLAB with functions in funcs_GraphEM.py

        This is optimization method for dealing with some object function hard to optim directly

        We dont need to explicitly compute Q(A, Ai)

        We simplify the question into computing PhiBCD... and do optim process directly

        """

        REG_TYPE = kwargs.get("reg_type")

        GAMMA = kwargs.get("gamma")
        LAMBDA = kwargs.get("lambda")
        A_n = kwargs.get("A")
        SIGMA = kwargs.get("Sigma")
        PHI = kwargs.get("Phi")
        C = kwargs.get("C")
        T = kwargs.get("T")
        Q_COV = kwargs.get("Q")
        XI = kwargs.get("xi")

        NUM_ITERATION = 2000

        REG_TERM = {
        "Laplace": F.L1_wrt_A,
        "Gaussian": F.Gaussian_Prior_wrt_A,
        "Block Laplace": F.Block_L1_wrt_A,
        "Laplace+Gaussian": F.L1_plus_Gaussian_Prior_wrt_A,
        }

        Q_list = []
        neg_ell_list = []

        obj_qunatities = {
            "reg_type": REG_TYPE,
            "A": A_n,
            "Q": self.Q,
            "Sigma": SIGMA,
            "Phi": PHI,
            "C": C,
            "T": T,
            "lambda": LAMBDA,
        }

        obj_Q = F.Q_wrt_A_given_An(**obj_qunatities) # Q(A(0), A(n))
        Q_list.append(obj_Q)

        temp = self.ParamsDict
        temp["A"] = A_n

        # self.logger.info("Before loglikelihood cal")

        loglikelihood = self.loglikelihood(Y=self.Y, **temp)

        # self.logger.info("After loglikelihood cal")

        neg_ell_list.append(-loglikelihood + REG_TERM[REG_TYPE](A=A_n))

        A_i = A_n
        for idx_iteration in range(NUM_ITERATION): # add new stop condition

            # prox gamma f2 at A_i
            prox_f2 = {
                "Laplace": F.prox_gamma_L1_wrt_Ai,
                "Gaussian": F.prox_gamma_L2_wrt_Ai,
                "Laplace+Gaussian": F.prox_gamma_L1_plus_L2_wrt_Ai,
                "Block Laplace": None,
            }
            prox_f2_quantities = {
                "A": A_i,
                "gamma": GAMMA * LAMBDA,
            }
            Y_i = prox_f2[REG_TYPE](A=A_i, gamma=GAMMA * LAMBDA)

            # note that here we use 2 * A - Y
            prox_f1 = F.prox_gamma_minus_Q_wrt_Ai
            prox_f1_quantities = {
                "A": 2 * Y_i - A_i,
                "C": C,
                "Phi": PHI,
                "Q": Q_COV,
                "gamma": GAMMA,
                "T": T,
            }
            Z_i = prox_f1(**prox_f1_quantities)

            # update hidden var
            A_i += 1 * (Z_i - Y_i) # A_i+1 = A_i + 1 * (Z_i - Y_i)

            obj_qunatities = {
                "reg_type": REG_TYPE,
                "A": Y_i, # change from A to Y
                "Q": self.Q,
                "Sigma": SIGMA,
                "Phi": PHI,
                "C": C,
                "T": T,
                "lambda": LAMBDA,
            }

            obj_Q = F.Q_wrt_A_given_An(**obj_qunatities) # Q(A(i+1), A(n))
            Q_list.append(obj_Q)

            temp = self.ParamsDict
            temp["A"] = Y_i # change from A to Y

            # self.logger.info("Before loglikelihood cal")

            loglikelihood = self.loglikelihood(Y=self.Y, **temp)

            # self.logger.info("After loglikelihood cal")

            # neg_ell_list.append(-loglikelihood)
            neg_ell_list.append(-loglikelihood + REG_TERM[REG_TYPE](A=Y_i))
    
            if idx_iteration+1 == NUM_ITERATION:
                self.logger.info(f"Douglas Rachford did not converge after iteration {idx_iteration+1}")

            # check stop condition
            if idx_iteration > 0 and np.abs(
                Q_list[-1] - Q_list[-2]
                ) <= XI: # Q(A(i+1), A(n))-Q(A(i), A(n))
                self.logger.info(f"Douglas Rachford converged after iteration {idx_iteration+1}")
                break

            # if idx_iteration > 0 and np.abs(
            #     neg_ell_list[-1] - neg_ell_list[-2]
            #     ) <= XI:
            #     self.logger.info(f"Douglas Rachford converged after iteration {idx_iteration+1}")
            #     break

        # A_n_plus_1 = A_i
        A_n_plus_1 = Y_i # change A to Y

        return A_n_plus_1

    def parameter_estimation(self, Y: np.ndarray = None, **kwargs):

        REG_TYPE = kwargs.get("reg_type", "Laplace")
        self.logger.info(f"Start GraphEM Algorithm with {REG_TYPE} regularization.")

        NUM_ITERATION = kwargs.get("num_iteration", 10)
        LAMBDA = kwargs.get("lambda", 50) # penalty/prior control parameter
        GAMMA = kwargs.get("gamma", 1) # regularization parameter
        EPS = kwargs.get("eps", 1e-5) # stop condition for Main Loop
        XI = kwargs.get("xi", 1e-5) # stop condition for M-Step

        REG_TERM = {
        "Laplace": F.L1_wrt_A,
        "Gaussian": F.Gaussian_Prior_wrt_A,
        "Block Laplace": F.Block_L1_wrt_A,
        "Laplace+Gaussian": F.L1_plus_Gaussian_Prior_wrt_A,
        }

        self.Y = Y

        Results = {}

        """
        
        init A
        
        """

        A = np.zeros_like(self.A)

        for i in range(len(A)):
            for j in range(len(A)):
                A[i, j] = 0.2 ** abs(i - j)

        # A = self.A + np.random.normal(scale=0.01, size=self.A.shape)

        U, S, VT = np.linalg.svd(A)
        S = np.minimum(S, 0.99)
        A = U @ np.diag(S) @ VT

        # max_singular_value = np.max(S)
        # coef = 0.99 / max_singular_value
        # A = coef * A # A(0)

        fnorm = np.linalg.norm(A - self.A, 'fro')

        temp = self.ParamsDict
        temp["A"] = A

        # self.logger.info("Before loglikelihood cal")

        loglikelihood = self.loglikelihood(Y=Y, **temp)

        # self.logger.info("After loglikelihood cal")

        T = len(Y)

        Results[f"A 0:{NUM_ITERATION}"] = [A]
        Results[f"||A - A_true||F 0:{NUM_ITERATION}"] = [fnorm]
        Results[f"-ell(A|Y, A_true) 0:{NUM_ITERATION}"] = [-loglikelihood + REG_TERM[REG_TYPE](A=A)]
        Results[f"GraphEM Q 0:{NUM_ITERATION}"] = []

        for idx in range(NUM_ITERATION):

            # self.logger.info(f"GraphEM Iteration {idx+1}")
            
            temp = self.ParamsDict
            temp["A"] = A

            # get quantities w.r.t. Q(A, A(n))
            quantities = self.quantities_from_Q(Y=Y, **temp)

            # unpack the result
            Sigma = quantities['Sigma']
            Phi = quantities['Phi']
            B = quantities["B"]
            C = quantities['C']
            D = quantities['D']

            # object func for last step, Q(A(n), A(n))
            obj_qunatities = {
                "reg_type": REG_TYPE,
                "A": A,
                "Q": self.Q,
                "Sigma": Sigma,
                "Phi": Phi,
                "C": C,
                "T": T,
                "lambda": LAMBDA,
            }

            obj_Q = F.Q_wrt_A_given_An(**obj_qunatities) # Q(A(n), A(n))
            Results[f"GraphEM Q 0:{NUM_ITERATION}"].append(obj_Q)

            # optim em
            M_step_quantities = {
                "reg_type": REG_TYPE,
                "A": A,
                "Q": self.Q,
                "Sigma": Sigma,
                "Phi": Phi,
                "C": C,
                "T": T,
                "gamma": GAMMA,
                "xi": XI,
                "lambda": LAMBDA,
            }
            A = self._Douglas_Rachford(**M_step_quantities) # A(n+1)

            fnorm = np.linalg.norm(A - self.A, 'fro')

            temp = self.ParamsDict
            temp["A"] = A # A(n+1)

            # self.logger.info("Before loglikelihood cal")

            loglikelihood = self.loglikelihood(Y=Y, **temp)

            # self.logger.info("After loglikelihood cal")

            Results[f"A 0:{NUM_ITERATION}"].append(A)
            Results[f"||A - A_true||F 0:{NUM_ITERATION}"].append(fnorm)
            Results[f"-ell(A|Y, A_true) 0:{NUM_ITERATION}"].append(-loglikelihood + REG_TERM[REG_TYPE](A=A))

            obj_qunatities = {
                "reg_type": REG_TYPE,
                "A": A,
                "Q": self.Q,
                "Sigma": Sigma,
                "Phi": Phi,
                "C": C,
                "T": T,
                "lambda": LAMBDA,
            }

            obj_Q = F.Q_wrt_A_given_An(**obj_qunatities) # Q(A(n+1), A(n))
            Results[f"GraphEM Q 0:{NUM_ITERATION}"].append(obj_Q)

            if idx+1 == NUM_ITERATION:
                self.logger.info(f"GraphEM did not converge after iteration {idx+1}")

            # check stop condition
            # if idx and np.abs(
            #     Results[f"GraphEM Q 0:{NUM_ITERATION}"][-1] - Results[f"GraphEM Q 0:{NUM_ITERATION}"][-2]
            #     ) <= EPS: # Q(A(n+1), A(n))-Q(A(n), A(n))
            #     self.logger.info(f"GraphEM converged after iteration {idx+1}")
            #     break

            if idx and np.abs(
                Results[f"-ell(A|Y, A_true) 0:{NUM_ITERATION}"][-1] - Results[f"-ell(A|Y, A_true) 0:{NUM_ITERATION}"][-2]
                ) <= EPS:
                self.logger.info(f"GraphEM converged after iteration {idx+1}")
                break

        Results[f"A"] = Results[f"A 0:{NUM_ITERATION}"]
        Results[f"A Fnorm"] = Results[f"||A - A_true||F 0:{NUM_ITERATION}"]
        Results[f"A NegLoglikelihood"] = Results[f"-ell(A|Y, A_true) 0:{NUM_ITERATION}"]
        # Results[f"A NegLoglikelihood"] = [L_t - REG_TERM[REG_TYPE](A=A) for L_t in Results[f"A NegLoglikelihood"]]
        Results[f"GraphEM Q Pair"] = Results[f"GraphEM Q 0:{NUM_ITERATION}"] # Q(A(0), A(0))-Q(A(1), A(0)), Q(A(1), A(1))-Q(A(2), A(1)), ...

        return Results
    

if __name__ == "__main__":

    from src.KalmanProcess import LinearGaussianDataGenerator, KalmanProcess

    np.random.seed(42)

    dim_x = 3
    MODELPARAMS = {
        "A": np.eye(dim_x) * 0.9,
        "Q": np.eye(dim_x) * 0.01,
        "H": np.eye(dim_x),
        "R": np.eye(dim_x) * 0.01,
        "m0": np.zeros(dim_x),
        "P0": np.eye(dim_x) * 0.0001,
    }

    LGSSM = KalmanProcess(**MODELPARAMS)
    LGSSM.generate_measurement(T=50)
    Y = LGSSM.Y

    print(f"-ell(A|Y): {-LGSSM.loglikelihood(Y=Y)}")

    ALG = GraphEMforA(**MODELPARAMS)
    GRAPH_EM_CONFIG = {
        "reg_type": "Laplace+Gaussian",
        "num_iteration": 100,
        "gamma": 1e-3, # Douglas-Rachford control parameter
        "lambda": 1, # penalty/prior control parameter
        "eps": 1e-5,
        "xi": 1e-5,
    }
    results = ALG.parameter_estimation(Y=Y, **GRAPH_EM_CONFIG)

    print(f"A: {results['A'][-1]}")
    print(f"A Fnorm: {results['A Fnorm'][-1]}")
    print(f"A -ell: {results['A NegLoglikelihood'][-1]}")

    pass
