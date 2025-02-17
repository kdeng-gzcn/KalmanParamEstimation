import sys
sys.path.append("./")

import numpy as np
import torch

from src.KalmanProcess import KalmanProcess

class GradientParameterEstimationAll(KalmanProcess):

    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None, var="A"):

        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0)

        # extract theta from vars
        self.theta = var

    def gradient_ell_k(self, y, A, H, Q, R, mu, P):

        """

        This func uses torch pkg for computing gradient.

        backword()

        Args:
            y: numpy array, the observation vector
            A: scalar (numpy), the variable with respect to which gradient will be computed
            mu_k-1: numpy array, prior mean
            P_k-1: 
            H: numpy array, transformation matrix
            Q: numpy array, process noise covariance
            R: numpy array, measurement noise covariance
        Returns:
            d ell_k: scalar since A is scalar

        """

        # to Tensor()
        # Convert inputs to torch tensors
        y = torch.tensor(y, dtype=torch.float32)

        A = torch.tensor(A, dtype=torch.float32)
        H = torch.tensor(H, dtype=torch.float32)
        mu = torch.tensor(mu, dtype=torch.float32)
        P = torch.tensor(P, dtype=torch.float32)
        Q = torch.tensor(Q, dtype=torch.float32)
        R = torch.tensor(R, dtype=torch.float32)

        # Ensure Theta is a torch tensor with gradient tracking
        if self.theta == "A":
            A.requires_grad = True
        elif self.theta == "H":
            H.requires_grad = True
        elif self.theta == "mu":
            mu.requires_grad = True
        elif self.theta == "P":
            P.requires_grad = True
        elif self.theta == "Q":
            Q.requires_grad = True
        elif self.theta == "R":
            R.requires_grad = True

        # # Function to compute mu and Sigma
        # def compute_mu_sigma():
        #     mean = H @ A @ mu
        #     Sigma = H @ A @ P @ A.T @ H.T + H @ Q @ H.T + R
        #     return mean, Sigma

        # # Compute mu and Sigma
        # mean, Sigma = compute_mu_sigma()

        mean = H @ A @ mu
        Sigma = H @ A @ P @ A.T @ H.T + H @ Q @ H.T + R

        mvn = torch.distributions.MultivariateNormal(mean, Sigma)

        # Compute f(mu, Sigma)
        f = mvn.log_prob(y)

        # Perform backward pass to compute gradients
        f.backward()

        # Extract the gradient with respect to A
        if self.theta == "A":
            gradient = A.grad
        elif self.theta == "H":
            gradient = H.grad
        elif self.theta == "mu":
            gradient =mu.grad
        elif self.theta == "P":
            gradient = P.grad
        elif self.theta == "Q":
            gradient = Q.grad
        elif self.theta == "R":
            gradient =R.grad

        # Return the gradient as a numpy scalar
        return gradient.cpu().numpy()
    
    def gradient_ell(self, Theta, Y=None):

        if Y is None:
            Y = self.Y

        # in each step k, use Filter in KalmanClass
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

        # return {'EX': self.Mu, 'Y': Y, 'P': self.Ps, 'M-': self.Mu_minus, 'P-': self.Ps_minus, 'V': self.V, 'S': self.Ss, 'K': self.Ks}
        result_dict = model.Filter(Y=Y)

        gradient_ell = []
        
        for idx, y in enumerate(Y):
            
            d_ell_k = self.gradient_ell_k(y, model.A, model.H, model.Sigma_q, model.Sigma_r, result_dict['EX'][idx], result_dict['P'][idx])

            gradient_ell.append(d_ell_k)

        gradient_ell = np.stack(gradient_ell, axis=0).sum(axis=0)

        return gradient_ell
    
    def parameter_estimation(self, alpha, Y=None, num_iteration=10):

        if Y is None:
            Y = self.Y

        if self.theta == "A":

            Theta = np.random.uniform(low=0., high=1., size=self.A.shape)
            m = np.linalg.norm(Theta - self.A, 'fro')

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
            Theta = np.random.uniform(low=0., high=0.02, size=self.Sigma_q.shape)
            m = np.linalg.norm(Theta - self.Sigma_q, 'fro')
        elif self.theta == "R":
            Theta = np.random.uniform(low=0., high=0.02, size=self.Sigma_r.shape)
            m = np.linalg.norm(Theta - self.Sigma_r, 'fro')

        print('Theta0:', Theta)
        print("metric0:", m)

        Thetas = [Theta]
        metric = [m]

        for _ in range(num_iteration):
                
            print("Gradient of ell:", self.gradient_ell(Theta, Y))
            Theta = Theta + alpha * self.gradient_ell(Theta, Y)

            if self.theta == "A":
                m = np.linalg.norm(Theta-self.A, 'fro')
            elif self.theta == "H": 
                m = np.linalg.norm(Theta-self.H, 'fro')
            elif self.theta == "mu":
                m = np.linalg.norm(Theta-self.mu_0, 2)
            elif self.theta == "P":
                m = np.linalg.norm(Theta-self.P_0, 'fro')
            elif self.theta == "Q":
                m = np.linalg.norm(Theta-self.Sigma_q, 'fro')
            elif self.theta == "R":
                m = np.linalg.norm(Theta-self.Sigma_r, 'fro')

            Thetas.append(Theta)
            metric.append(m)

        return Theta, Thetas, metric
    
class GradientParameterEstimationA(KalmanProcess):
    """
    High-Level Usage: input Y and output a fitted model. Remember we have build-in data for evaluation.

    To compute the gradient of ell in closed-form, we need to use quantities from Filter part. 
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def gradient_ell_k(self, y, A, mu, P, H, Q, R):
        """
        This func uses torch pkg for computing gradient.

        backword()

        Args:
            y: numpy array, the observation vector
            A: scalar (numpy), the variable with respect to which gradient will be computed
            mu_k-1: numpy array, prior mean
            P_k-1: 
            H: numpy array, transformation matrix
            Q: numpy array, process noise covariance
            R: numpy array, measurement noise covariance
        Returns:
            d ell_k: scalar since A is scalar
        """

        # to Tensor()
        # Convert inputs to torch tensors
        y = torch.tensor(y, dtype=torch.float32)
        H = torch.tensor(H, dtype=torch.float32)
        mu = torch.tensor(mu, dtype=torch.float32)
        P = torch.tensor(P, dtype=torch.float32)
        Q = torch.tensor(Q, dtype=torch.float32)
        R = torch.tensor(R, dtype=torch.float32)

        # Ensure A is a torch tensor with gradient tracking
        A = torch.tensor(A, dtype=torch.float32, requires_grad=True)

        # Function to compute mu and Sigma
        def compute_mu_sigma(A):
            mean = H @ A @ mu
            Sigma = H @ A @ P @ A.T @ H.T + H @ Q @ H.T + R
            return mean, Sigma

        # Compute mu and Sigma
        mean, Sigma = compute_mu_sigma(A)

        mvn = torch.distributions.MultivariateNormal(mean, Sigma)

        # Compute f(mu, Sigma)
        f = mvn.log_prob(y)

        # Perform backward pass to compute gradients
        f.backward()

        # Extract the gradient with respect to A
        gradient_A = A.grad

        # Return the gradient as a numpy scalar
        return gradient_A.cpu().numpy()

    def gradient_ell(self, A, Y=None):
        if Y is None:
            Y = self.Y

        # in each step k, use Filter in KalmanClass
        model = KalmanProcess(A=A, Sigma_q=self.Sigma_q, H=self.H, 
                                    Sigma_r=self.Sigma_r, mu_0=self.mu_0, P_0=self.P_0)
        # return {'EX': self.Mu, 'Y': Y, 'P': self.Ps, 'M-': self.Mu_minus, 'P-': self.Ps_minus, 'V': self.V, 'S': self.Ss, 'K': self.Ks}
        result_dict = model.Filter(Y=Y)

        gradient_ell = []
        
        for idx, y in enumerate(Y):
            # print(result_dict['EX'][idx])
            d_ell_k = self.gradient_ell_k(y, A, result_dict['EX'][idx], result_dict['P'][idx], self.H, self.Sigma_q, self.Sigma_r)

            gradient_ell.append(d_ell_k)

        gradient_ell = np.stack(gradient_ell, axis=0).sum(axis=0)

        return gradient_ell
    
    def numerical_gradient_ell(self, A, Y):

        epsilon = np.full(A.shape, 1e-6)
    
        f_A_plus_eps = self.loglikelihood(A + epsilon, Y)
        f_A_minus_eps = self.loglikelihood(A - epsilon, Y)
        
        grad_A = (f_A_plus_eps - f_A_minus_eps) / (2 * epsilon)

        return grad_A

    def parameter_estimation(self, alpha, Y=None, num_iteration=10, numerical=False):

        if Y is None:
            Y = self.Y

        dimx = len(self.A)

        # init value, make sure A is > 0
        A = np.random.randn(dimx, dimx)

        print('A0', A)

        As = [A]
        metric = [np.linalg.norm(A-self.A, 'fro')]

        for _ in range(num_iteration):

            if numerical:
                print("dell", self.numerical_gradient_ell(A, Y))
                A = A + alpha * self.numerical_gradient_ell(A, Y)
                m = np.linalg.norm(A-self.A, 'fro')
            else:
                print("dell", self.gradient_ell(A, Y))
                A = A + alpha * self.gradient_ell(A, Y)
                m = np.linalg.norm(A-self.A, 'fro')

            As.append(A)
            metric.append(m)

        return A, As, metric
    
if __name__ == "__main__":

    pass
    