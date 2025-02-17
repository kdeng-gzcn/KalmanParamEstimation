import numpy as np

class LinearGaussianDataGenerator:

    def __init__(self, **kwargs):

        # 1
        self.A = kwargs.get("A")
        self.Q = kwargs.get("Q")
        self.H = kwargs.get("H")
        self.R = kwargs.get("R")
        self.mu_0 = kwargs.get("m0")
        self.P_0 = kwargs.get("P0")
        
        # 2
        self.ParamsDict = {
            "A": self.A,
            "H": self.H,
            "Q": self.Q,
            "R": self.R,
            "m0": self.mu_0,
            "P0": self.P_0,
        }

    def _dynamic(self, x_k: np.ndarray):

        q_k = np.random.multivariate_normal(
            mean=np.zeros(shape=x_k.shape), 
            cov=self.Q,
        )

        x_k_plus_1 = self.A @ x_k + q_k

        return x_k_plus_1

    def _measurement(self, x_k: np.ndarray):
        
        r_k = np.random.multivariate_normal(
            mean=np.zeros(shape=x_k.shape), 
            cov=self.R,
        )

        y_k = self.H @ x_k + r_k

        return y_k
    
    def generate_measurement(self, T: int):

        try: 

            self.T = T

            x_0 = np.random.multivariate_normal(
                mean=self.mu_0, 
                cov=self.P_0,
            )

            self.X = np.zeros(shape=(self.T + 1, self.A.shape[0])) # T*n
            self.X[0, :] = x_0
            self.Y = np.zeros(shape=(self.T, self.H.shape[0])) # T*m

            x_k = x_0

            for k in range(self.T):

                x_k = self._dynamic(x_k)
                y_k = self._measurement(x_k)

                self.X[k+1, :] = x_k
                self.Y[k, :] = y_k

            # 3
            self.DataDict = {
                "X 0:T": self.X,
                "Y 1:T": self.Y
            }

            return self.DataDict
        
        except Exception as e:

            print(f"Error in Data Generation: {e}")
    