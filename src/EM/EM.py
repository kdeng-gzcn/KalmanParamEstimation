from .KalmanClass import KalmanClass
import numpy as np
import torch

class EMParameterEstimationAll(KalmanClass):

    """

    High-Level Usage: input Y and output a fitted model. Remember we have build-in data for evaluation.

    To compute the conclusions from EM Algorithm, we need to use quantities from Filter and Smoother part.

    """

    def __init__(self, var: str, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None) -> None:
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0) # use default value

        self.theta = var

    def quantities_from_Q(self, Theta, Y=None):

        if Y is None:
            Y = self.Y

        model = KalmanClass(A=self.A, Sigma_q=self.Sigma_q, H=self.H, 
                                    Sigma_r=self.Sigma_r, mu_0=self.mu_0, P_0=self.P_0)
        
        if self.theta == "A":
            model.A = Theta
        elif self.theta == "H":
            model.H = Theta
        elif self.theta == "mu":
            model.mu_0 = Theta
        elif self.theta == "P":
            model.P_0 = Theta
        elif self.theta == "Q":
            model.Sigma_q = Theta
        elif self.theta == "R":
            model.Sigma_r = Theta
        
        model.Filter(Y=Y)
        # return {'EX Smoother': self.Mu_Smoother, 'P Smoother': self.Ps_Smoother, 'G': self.Gs}
        smoother_dict = model.Smoother(Y=Y)

        Mu_Smoother = smoother_dict["EX Smoother"]
        Ps_Smoother = smoother_dict["P Smoother"]
        Gs = smoother_dict["G"]

        T = len(Y)

        # idx = range(1, 51)

        # init
        Sigma = np.zeros_like(Ps_Smoother[0])
        Phi = np.zeros_like(Ps_Smoother[0])
        B = np.zeros((Y[0].shape[0], Mu_Smoother[0].shape[0]))
        C = np.zeros_like(Ps_Smoother[0])
        D = np.zeros((Y[0].shape[0], Y[0].shape[0]))

        for k in range(1, T+1):

            # Sigma: Σ = (1/T) * Σ_{k=1}^{T} (P_s^k + m_s^k * (m_s^k)^T)
            Sigma += Ps_Smoother[k] + np.outer(Mu_Smoother[k], Mu_Smoother[k])

            # Phi: Φ = (1/T) * Σ_{k=1}^{T} (P_s^{k-1} + m_s^{k-1} * (m_s^{k-1})^T)
            Phi += Ps_Smoother[k-1] + np.outer(Mu_Smoother[k-1], Mu_Smoother[k-1])

            # B: B = (1/T) * Σ_{k=1}^{T} (y_k * (m_s^k)^T)
            B += np.outer(Y[k-1], Mu_Smoother[k])

            # C: C = (1/T) * Σ_{k=1}^{T} (P_s^{k-1} G^T_k + m_s^k * (m_s^{k-1})^T)
            C += Ps_Smoother[k] @ Gs[k-1].T + np.outer(Mu_Smoother[k], Mu_Smoother[k-1])

            # D: D = (1/T) * Σ_{k=1}^{T} (y_k * y_k^T)
            D += np.outer(Y[k-1], Y[k-1])

        Sigma /= T
        Phi /= T
        B /= T
        C /= T
        D /= T

        return {"Sigma": Sigma, "Phi": Phi, "B": B, "C": C, "D": D, "EX Smoother": Mu_Smoother, "P Smoother": Ps_Smoother}

    def parameter_estimation(self, Y=None, num_iteration=10):

        if Y is None:
            Y = self.Y

        if self.theta == "A":

            # Theta = np.random.uniform(low=0., high=1., size=self.A.shape)

            Theta = np.zeros_like(self.A)

            for i in range(len(Theta)):
                for j in range(len(Theta)):
                    Theta[i, j] = 0.1 ** abs(i - j)

            U, S, VT = np.linalg.svd(Theta)
            max_singular_value = np.max(S)
            coef = 0.99 / max_singular_value
            Theta = coef * Theta

            m = np.linalg.norm(Theta - self.A, 'fro')
            loglike = self.loglikelihood(theta=Theta, Y=Y)

        elif self.theta == "H":
            Theta = np.random.uniform(low=0., high=1., size=self.H.shape)
            m = np.linalg.norm(Theta - self.H, 'fro')
        elif self.theta == "mu":
            Theta = np.random.normal(loc=0., scale=1., size=self.mu_0.shape)
            m = np.linalg.norm(Theta - self.mu_0, 2)
        elif self.theta == "P":
            Theta = np.random.uniform(low=0., high=0.02, size=self.P_0.shape)
            m = np.linalg.norm(Theta - self.P_0, 'fro')
        elif self.theta == "Q":
            Theta = np.random.uniform(low=0., high=0.5, size=self.Sigma_q.shape)
            Theta = Theta @ Theta.T # symmetric Cov Mat
            m = np.linalg.norm(Theta - self.Sigma_q, 'fro')
            loglike = self.loglikelihood(theta=Theta, Y=Y)
        elif self.theta == "R":
            Theta = np.random.uniform(low=0., high=0.02, size=self.Sigma_r.shape)
            m = np.linalg.norm(Theta - self.Sigma_r, 'fro')

        # print('Theta0:', Theta)
        # print("metric0:", m)

        Thetas = [Theta]
        metric = [m]
        neg_loglikelihood_list = [-loglike]

        for _ in range(num_iteration):

            result = self.quantities_from_Q(Theta=Theta, Y=Y)
            
            if self.theta == "A":
                # update A = C Phi-1
                Theta = result["C"] @ np.linalg.inv(result["Phi"])
                m = np.linalg.norm(Theta-self.A, 'fro')
                loglike = self.loglikelihood(theta=Theta, Y=Y)
            elif self.theta == "H":
                # H = B Simga-1
                Theta = result["B"] @ np.linalg.inv(result["Sigma"])
                m = np.linalg.norm(Theta-self.H, 'fro')
            elif self.theta == "mu":
                # m = m_0_Smoother
                Theta = result["EX Smoother"][0]
                m = np.linalg.norm(Theta-self.mu_0, 2)
            elif self.theta == "P":
                # P = P_0_Smoother + (m_0_Smoother - m_0)(m_0_Smoother - m_0)^T
                Theta = result["P Smoother"][0] + (result["EX Smoother"][0] - self.mu_0) @ (result["EX Smoother"][0] - self.mu_0).T
                m = np.linalg.norm(Theta-self.P_0, 'fro')
            elif self.theta == "Q":
                # Q = Sigma - C A^T - A C^T + A Phi A^T 
                Theta = result["Sigma"] - result["C"] @ self.A.T - self.A @ result["C"].T + self.A @ result["Phi"] @ self.A.T
                m = np.linalg.norm(Theta-self.Sigma_q, 'fro')
            elif self.theta == "R":
                # R = D - H B^T - B H^T + H Sigma H^T
                Theta = result["D"] - self.H @ result["B"].T - result["B"] @ self.H.T + self.H @ result["Sigma"] @ self.H.T
                m = np.linalg.norm(Theta-self.Sigma_r, 'fro')

            Thetas.append(Theta)
            metric.append(m)
            neg_loglikelihood_list.append(-loglike)

        fnorm_list = metric

        return Theta, Thetas, fnorm_list, neg_loglikelihood_list
    