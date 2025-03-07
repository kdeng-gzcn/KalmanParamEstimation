# for local import
# import sys
# sys.path.append("")
import logging

import numpy as np
from pykalman import KalmanFilter as KFpykalman
from filterpy.kalman import KalmanFilter as KFfilterpy

from src.DataGenerator import LinearGaussianDataGenerator
from src.logging.logging_config import setup_logging

class KalmanProcess(LinearGaussianDataGenerator):

    """

    This class is for Filter and Smoother given measurements. Hence it heritages the data from Generator Class.

    filter:
        need model params and initial info (may be unknown) and Y (must be known) to estimate E X|y1:k, Var X|y1:k

    smoother:
        based on filter's results, backward compute E X|Y, Var X|Y
        
    """

    def __init__(self, **kwargs):

        self.logger = logging.getLogger(__name__)

        super().__init__(**kwargs) # use default value

        # KF from pykalman for loglikelihood part
        self.kf = KFpykalman(
            transition_matrices=self.A,
            transition_covariance=self.Q,
            observation_matrices=self.H,
            observation_covariance=self.R,
            initial_state_mean=self.mu_0,
            initial_state_covariance=self.P_0
        )

        # KF from filterpy for filter step
        self.kf2 = KFfilterpy(dim_x=self.A.shape[0], dim_z=self.H.shape[0])
        self.kf2.x = self.mu_0
        self.kf2.F = self.A
        self.kf2.H = self.H
        self.kf2.R = self.R
        self.kf2.Q = self.Q
        self.kf2.P = self.P_0
    
    def Filter(self, Y: np.ndarray = None):

        """

        This function is to complete the Filter in detail.
        Create the collections of midterm quantity during Filter.

        We use `filterpy` here because it provides Prediction and Update step.

        """
            
        self.Y = Y
        self.T, _ = Y.shape

        # Note that Y is measurement matrix
        self.FilterDict = {}

        self.FilterDict["m- 1:T"] = np.zeros(shape=(self.T, self.A.shape[0]))
        self.FilterDict["P- 1:T"] = np.zeros(shape=(self.T, *self.P_0.shape))

        self.FilterDict["v 1:T"] = np.zeros(shape=(self.T, len(self.H)))
        self.FilterDict["S 1:T"] = np.zeros(shape=(self.T, *self.R.shape))
        self.FilterDict["K 1:T"] = np.zeros(shape=(self.T, *self.H.T.shape))
        self.FilterDict["m 0:T"] = np.zeros(shape=(self.T + 1, self.A.shape[0]))
        self.FilterDict["P 0:T"] = np.zeros(shape=(self.T + 1, *self.P_0.shape))

        self.FilterDict["m 0:T"][0, :] = self.mu_0
        self.FilterDict["P 0:T"][0, :, :] = self.P_0

        # forloop T times
        for k, y in enumerate(Y): # iterate each row in Y

            # prediction
            m_ = self.A @ self.FilterDict["m 0:T"][k]
            P_ = self.A @ self.FilterDict["P 0:T"][k] @ self.A.T + self.Q

            self.FilterDict["m- 1:T"][k, :] = m_
            self.FilterDict["P- 1:T"][k, :, :] = P_

            # update
            v = y - self.H @ m_
            S = self.H @ P_ @ self.H.T + self.R
            K = P_ @ self.H.T @ np.linalg.inv(S)
            m = m_ + K @ v
            P = P_ - K @ S @ K.T

            self.FilterDict["v 1:T"][k, :] = v
            self.FilterDict["S 1:T"][k, :, :] = S
            self.FilterDict["K 1:T"][k, :, :] = K
            self.FilterDict["m 0:T"][k+1, :] = m
            self.FilterDict["P 0:T"][k+1, :, :] = P

        return self.FilterDict
    
    def Smoother(self, Y: np.ndarray = None):

        """

        This function is to complete the Smoother given measurement and corresponding Filter quantities.

        It smooths the estimates of the filtered values by iterating backward.

        """
            
        self.Y = Y

        self.Filter(Y=self.Y)

        Mu = self.FilterDict["m 0:T"]
        Ps = self.FilterDict["P 0:T"]

        Mu_minus = self.FilterDict.get("m- 1:T")
        Ps_minus = self.FilterDict.get("P- 1:T")

        self.SmootherDict = {}

        # Initialize the smoother arrays
        self.SmootherDict["G 0:T-1"] = np.zeros((self.T, *self.A.shape))  # Gain matrix for each time step
        self.SmootherDict["m smo 0:T"] = np.zeros((self.T + 1, len(self.A)))  # Smoothed state estimates
        self.SmootherDict["P smo 0:T"] = np.zeros((self.T + 1, *self.P_0.shape))  # Smoothed covariance estimates

        # Backward iteration to calculate the smoothed estimates
        self.SmootherDict["m smo 0:T"][-1, :] = Mu[-1, :]
        self.SmootherDict["P smo 0:T"][-1, :, :] = Ps[-1, :, :]

        # Compute G_k and the smoothed estimates
        for k in reversed(range(self.T)):  # backward from T-1 to 0

            G = Ps[k] @ self.A.T @ np.linalg.inv(Ps_minus[k])
            m_smo = Mu[k] + G @ (self.SmootherDict["m smo 0:T"][k+1] - Mu_minus[k])
            P_smo = Ps[k] + G @ (self.SmootherDict["P smo 0:T"][k+1] - Ps_minus[k]) @ G.T

            self.SmootherDict["G 0:T-1"][k, :, :] = G
            self.SmootherDict["m smo 0:T"][k, :] = m_smo
            self.SmootherDict["P smo 0:T"][k, :, :] = P_smo

        return self.SmootherDict
    
    def loglikelihood(self, Y: np.ndarray = None, **kwargs):

        self.Y = Y

        # if kwargs:
        #     model = KalmanProcess(**kwargs)
        #     filter = model.Filter(Y=Y)
        # else:
        #     filter = self.Filter(Y=Y)

        model = KalmanProcess(**kwargs)
        filter = model.Filter(Y=Y)

        T, p = Y.shape

        ell = 0

        for k in range(T): # 0:T-1

            vk = filter["v 1:T"][k]
            Sk = filter["S 1:T"][k]

            ell += -(1/2 * np.log(2 * np.pi * np.linalg.det(Sk)) + 1/2 * vk.T @ np.linalg.inv(Sk) @ vk)

        return ell
    
    def quantities_from_Q(self, Y: np.ndarray = None, **kwargs):

        self.Y = Y

        if kwargs:
            model = KalmanProcess(**kwargs)
            smoother = model.Smoother(Y=Y)
        else:
            smoother = self.Smoother(Y=Y)

        T, _ = Y.shape

        # init
        Sigma = np.zeros_like(smoother["P smo 0:T"][0])
        Phi = np.zeros_like(smoother["P smo 0:T"][0])
        B = np.zeros((Y[0].shape[0], smoother["m smo 0:T"][0].shape[0]))
        C = np.zeros_like(smoother["P smo 0:T"][0])
        D = np.zeros((Y[0].shape[0], Y[0].shape[0]))

        for k in range(1, T+1):

            # Sigma: Σ = (1/T) * Σ_{k=1}^{T} (P_s^k + m_s^k * (m_s^k)^T)
            Sigma += smoother["P smo 0:T"][k] + np.outer(smoother["m smo 0:T"][k], smoother["m smo 0:T"][k])

            # Phi: Φ = (1/T) * Σ_{k=1}^{T} (P_s^{k-1} + m_s^{k-1} * (m_s^{k-1})^T)
            Phi += smoother["P smo 0:T"][k-1] + np.outer(smoother["m smo 0:T"][k-1], smoother["m smo 0:T"][k-1])

            # B: B = (1/T) * Σ_{k=1}^{T} (y_k * (m_s^k)^T)
            B += np.outer(Y[k-1], smoother["m smo 0:T"][k])

            # C: C = (1/T) * Σ_{k=1}^{T} (P_s^{k-1} G^T_k + m_s^k * (m_s^{k-1})^T)
            C += smoother["P smo 0:T"][k] @ smoother["G 0:T-1"][k-1].T + np.outer(smoother["m smo 0:T"][k], smoother["m smo 0:T"][k-1])

            # D: D = (1/T) * Σ_{k=1}^{T} (y_k * y_k^T)
            D += np.outer(Y[k-1], Y[k-1])

        Sigma /= T
        Phi /= T
        B /= T
        C /= T
        D /= T

        self.quantities = {
            "Sigma": Sigma, 
            "Phi": Phi, 
            "B": B, 
            "C": C, 
            "D": D,
            "m smo 0:T": smoother["m smo 0:T"],
            "P smo 0:T": smoother["P smo 0:T"],
        }

        return self.quantities

if __name__ == "__main__":

    np.random.seed(42)

    dim_x = 16
    A = np.eye(dim_x) * 0.9  # Initial A matrix, scaled identity matrix
    Q = np.eye(dim_x) * 0.01  # Small noise in Sigma_q
    H = np.eye(dim_x)  # H matrix as identity
    R = np.eye(dim_x) * 0.01  # Small noise in Sigma_r
    m0 = np.zeros(dim_x)  # Zero vector for mu_0
    P0 = np.eye(dim_x) * 0.01  # Small values in P_0

    ModelParams = {
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
        "m0": m0,
        "P0": P0,
    }

    datamodel = LinearGaussianDataGenerator(**ModelParams)
    data = datamodel.generate_measurement(T=50)

    Y = datamodel.Y
    # Y = data["Y 1:T"]

    model = KalmanProcess(**ModelParams)

    filter = model.Filter(Y=Y)
    smoother = model.Smoother(Y=Y)

    ell = model.loglikelihood(Y=Y)

    print(ell)

    TestParams = {
        "A": np.eye(dim_x) * 0.5,
        "Q": Q,
        "H": H,
        "R": R,
        "m0": m0,
        "P0": P0,
    }

    model = KalmanProcess(**TestParams)
    ell = model.loglikelihood(Y=data["Y 1:T"])

    print(ell)

    # print(model.quantities_from_Q(Y=Y))
    
    pass
