# Extended Regularization Strategies for GraphEM under Graphical Inference

## Usage

### GraphEM

~~~python
import numpy as np
from src.KalmanProcess import KalmanProcess
from src.GraphEM import GraphEMforA

np.random.seed(42)

dim_x = 8
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
    "gamma": 1, # Douglas-Rachford control parameter
    "lambda": 1, # penalty/prior control parameter
    "eps": 1e-5,
    "xi": 1e-5,
}
results = ALG.parameter_estimation(Y=Y, **GRAPH_EM_CONFIG)

print(f"A: {results['A'][-1]}")
print(f"A Fnorm: {results['A Fnorm'][-1]}")
print(f"A -ell: {results['A NegLoglikelihood'][-1]}")
~~~

### Kalman Filter and Smoother

~~~python
import numpy as np
from src.DataGenerator import LinearGaussianDataGenerator
from src.KalmanProcess import KalmanProcess

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
~~~

### EM Algorithm

~~~python
import numpy as np
from src.KalmanProcess import LinearGaussianDataGenerator
from src.EM import EMParameterEstimation

np.random.seed(42)

dim_x = 8
ModelParams = {
    "A": np.eye(dim_x) * 0.9,
    "Q": np.eye(dim_x) * 0.01,
    "H": np.eye(dim_x),
    "R": np.eye(dim_x) * 0.01,
    "m0": np.zeros(dim_x),
    "P0": np.eye(dim_x) * 0.0001,
}

LGSSM = LinearGaussianDataGenerator(**ModelParams)
LGSSM.generate_measurement(T=50)
Y = LGSSM.Y

alg = EMParameterEstimation(**ModelParams)
missing_vars = ["A", "H", "Q", "R", "m0", "P0"]
results = alg.parameter_estimation(missing_vars=missing_vars, Y=Y)

for var in missing_vars:

    print(f"{var}: {results[var][-1]}")
    print(f"{var} Fnorm: {results[var + ' Fnorm'][-1]}")
    print(f"{var} -ell: {results[var + ' NegLoglikelihood'][-1]}")
    print()
~~~
