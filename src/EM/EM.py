import sys
sys.path.append("")
from typing import List

import numpy as np

from src.KalmanProcess import KalmanProcess

class EMParameterEstimation(KalmanProcess):

    """

    High-Level Usage: input Y and output a fitted model. Remember we have build-in data for evaluation.

    To compute the conclusions from EM Algorithm, we need to use quantities from Filter and Smoother part.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def parameter_estimation(self, missing_vars: List[str], Y: np.ndarray = None, num_iteration: int = 10):

        self.Y = Y

        mapping = {
            "A": self.A,
            "Q": self.Q,
            "H": self.H,
            "R": self.R,
            "m0": self.mu_0,
            "P0": self.P_0,
        }

        Results = {}

        for var in missing_vars:

            print(f"Estimating Missing Value: {var}")

            Theta = np.zeros_like(mapping[var])

            # initialization not for vector m0
            for i in range(len(Theta)):
                for j in range(len(Theta)):
                    Theta[i, j] = 0.1 ** abs(i - j)

            U, S, VT = np.linalg.svd(Theta)
            max_singular_value = np.max(S)
            coef = 0.99 / max_singular_value
            Theta = coef * Theta

            fnorm = np.linalg.norm(Theta - mapping[var], 'fro')

            temp = self.ParamsDict
            temp[var] = Theta

            loglike = self.loglikelihood(Y=Y, **temp)

            Results[f"{var} 0:{num_iteration}"] = [Theta]
            Results[f"||{var} - {var}_true||F 0:{num_iteration}"] = [fnorm]
            Results[f"-ell({var}|Y, {var}_true) 0:{num_iteration}"] = [-loglike]

            for _ in range(num_iteration):

                temp = self.ParamsDict
                temp[var] = Theta

                quantities = self.quantities_from_Q(Y=Y, **temp)

                formula_mapping = {
                    "A": quantities["C"] @ np.linalg.inv(quantities["Phi"]),
                    "H": quantities["B"] @ np.linalg.inv(quantities["Sigma"]),
                    "m0": quantities["m smo 0:T"][0],
                    "P0": quantities["P smo 0:T"][0] + (quantities["m smo 0:T"][0] - self.mu_0) @ (quantities["m smo 0:T"][0] - self.mu_0).T,
                    "Q": quantities["Sigma"] - quantities["C"] @ self.A.T - self.A @ quantities["C"].T + self.A @ quantities["Phi"] @ self.A.T,
                    "R": quantities["D"] - self.H @ quantities["B"].T - quantities["B"] @ self.H.T + self.H @ quantities["Sigma"] @ self.H.T,
                }

                Theta = formula_mapping[var]

                fnorm = np.linalg.norm(Theta - mapping[var], 'fro')

                temp = self.ParamsDict
                temp[var] = Theta
                loglike = self.loglikelihood(Y=Y, **temp)

                Results[f"{var} 0:{num_iteration}"].append(Theta)
                Results[f"||{var} - {var}_true||F 0:{num_iteration}"].append(fnorm)
                Results[f"-ell({var}|Y, {var}_true) 0:{num_iteration}"].append(-loglike)

            Results[f"{var}"] = Results[f"{var} 0:{num_iteration}"]
            Results[f"{var} Fnorm"] = Results[f"||{var} - {var}_true||F 0:{num_iteration}"]
            Results[f"{var} NegLoglikelihood"] = Results[f"-ell({var}|Y, {var}_true) 0:{num_iteration}"]

        return Results
    
if __name__ == "__main__":

    from src.KalmanProcess import LinearGaussianDataGenerator

    np.random.seed(42)

    dim_x = 3
    ModelParams = {
        "A": np.eye(dim_x) * 0.9,
        "Q": np.eye(dim_x) * 0.01,
        "H": np.eye(dim_x),
        "R": np.eye(dim_x) * 0.01,
        "m0": np.zeros(dim_x),
        "P0": np.eye(dim_x) * 0.01,
    }

    ssm = LinearGaussianDataGenerator(**ModelParams)
    ssm.generate_measurement(T=50)
    Y = ssm.Y

    alg = EMParameterEstimation(**ModelParams)
    missing_vars = ["A", "H"]
    results = alg.parameter_estimation(missing_vars=missing_vars, Y=Y)

    for var in missing_vars:

        print(f"{var}: {results[var][-1]}")
        print(f"{var} Fnorm: {results[var + ' Fnorm'][-1]}")
        print(f"{var} -ell: {results[var + ' NegLoglikelihood'][-1]}")
        print()

    pass
