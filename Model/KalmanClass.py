import numpy as np
from matplotlib import pyplot as plt
import torch

from pykalman import KalmanFilter
from filterpy.kalman import KalmanFilter as KFfilterpy

class LinearGaussianDataGenerator:
    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None):

        if A is None:
            A = np.array([[1.]])
        if Sigma_q is None:
            Sigma_q = np.diag([0.01])
        if H is None:
            H = np.array([[1.]])
        if Sigma_r is None:
            Sigma_r = np.diag([0.01])
        if mu_0 is None:
            mu_0 = np.array([0])
        if P_0 is None:
            P_0 = np.diag([0.01])

        # init params
        self.A = A
        self.Sigma_q = Sigma_q
        self.H = H
        self.Sigma_r = Sigma_r
        self.mu_0 = mu_0
        self.P_0 = P_0

        # Default Data
        result = self.generate_measurement(total_timesteps=50)
        self.X = result['X']
        self.Y = result['Y']

    def dynamic(self, x_k):
        """
        x_k: (dim_x, )
        """
        q_k = np.random.multivariate_normal(mean=np.zeros(len(x_k)), cov=self.Sigma_q)
        x_k_plus_1 = np.dot(self.A, x_k) + q_k
        return x_k_plus_1

    def measurement(self, x_k):
        """
        x_k: (dim_x, )
        """
        r_k = np.random.multivariate_normal(mean=np.zeros(len(x_k)), cov=self.Sigma_r)
        y_k = np.dot(self.H, x_k) + r_k
        return y_k
    
    def generate_measurement(self, total_timesteps=50):
        """
        return:
            X: (t, dim_x)
            Y: (t, dim_y)
        """

        # (dim_x, )
        x_0 = np.random.multivariate_normal(mean=self.mu_0, cov=self.P_0)

        self.X = [x_0] # t*n
        self.Y = [] # t*m

        ### for loop
        x_k = x_0
        for _ in range(total_timesteps):

            x_k = self.dynamic(x_k)
            y_k = self.measurement(x_k)

            self.X.append(x_k)
            self.Y.append(y_k)

        return {'X': np.stack(self.X, axis=0), 'Y': np.stack(self.Y, axis=0)}

class KalmanClass(LinearGaussianDataGenerator):

    """

    This class is for Filter and Smoother given measurements. Hence it heritages the data from Generator Class.

    filter:
        need model params and initial info (may be unknown) and Y (must be known) to estimate E X|y1:k, Var X|y1:k

    smoother:
        based on filter's results, backward compute E X|Y, Var X|Y
        
    """

    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None) -> None:
        """
        we already have build-in X and Y
        """
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0) # use default value

        if A is not None:
            self.A = A
        if Sigma_q is not None:
            self.Sigma_q = Sigma_q
        if H is not None:
            self.H = H
        if Sigma_r is not None:
            self.Sigma_r = Sigma_r
        if mu_0 is not None:
            self.mu_0 = mu_0
        if P_0 is not None:
            self.P_0 = P_0

        # KF
        self.kf = KalmanFilter(
            transition_matrices=self.A,
            transition_covariance=self.Sigma_q,
            observation_matrices=self.H,
            observation_covariance=self.Sigma_r,
            initial_state_mean=self.mu_0,
            initial_state_covariance=self.P_0
        )

        # KF from filterpy
        self.kf2 = KFfilterpy(dim_x=len(self.A), dim_z=len(self.H))
        self.kf2.x = self.mu_0
        self.kf2.F = self.A
        self.kf2.H = self.H
        self.kf2.R = self.Sigma_r
        self.kf2.Q = self.Sigma_q
        self.kf2.P = self.P_0
    
    def Filter(self, Y=None):
        """
        This function is to complete the Filter in detail.
        Create the collections of midterm quantity during Filter.

        We use `filterpy` here because it provides Prediction and Update step.
        """

        # Use build-in data
        # (t, dim_y)
        if Y is None:
            Y = self.Y

        # Note that Y is measurement matrix
        # store mu_k P_k in each step

        self.Mu = [self.kf2.x]
        self.Ps =[self.kf2.P]
        self.Mu_minus = []
        self.Ps_minus = []
        self.V = []
        self.Ss = []
        self.Ks = []

        # forloop t times
        for y in Y: # iterate each column in Y

            # get mu|Y, Sigma|Y

            # self.mu_k, self.P_k = self.kf.filter_update(
            #         filtered_state_mean=self.mu_k, 
            #         filtered_state_covariance=self.P_k, 
            #         observation=y
            #         )

            # Prediction
            self.kf2.predict()

            self.Mu_minus.append(self.kf2.x)
            self.Ps_minus.append(self.kf2.P)

            self.v_k = y - self.H @ self.kf2.x
            self.S_k = self.H @ self.kf2.P @ self.H.T + self.Sigma_r

            # Update
            self.kf2.update(y)

            self.Mu.append(self.kf2.x)
            self.Ps.append(self.kf2.P)
            self.V.append(self.v_k)
            self.Ss.append(self.S_k)
            self.Ks.append(self.kf2.K)
            

        # list -> ndarray
        self.Mu = np.stack(self.Mu, axis=0)
        self.Ps = np.stack(self.Ps, axis=0)
        self.Mu_minus = np.stack(self.Mu_minus, axis=0)
        self.Ps_minus = np.stack(self.Ps_minus, axis=0)
        self.V = np.stack(self.V, axis=0)
        self.Ss = np.stack(self.Ss, axis=0)
        self.Ks = np.stack(self.Ks, axis=0)

        return {'EX': self.Mu, 'Y': Y, 'P': self.Ps, 'M-': self.Mu_minus, 'P-': self.Ps_minus, 'V': self.V, 'S': self.Ss, 'K': self.Ks}
    
    def Smoother(self, Y=None, Mu=None, Ps=None, Mu_minus=None, Ps_minus=None):
        """
        This function is to complete the Smoother given measurement and corredponding Filter quantities.
        """
        try:
            # Use build-in data
            if Y is None:
                Y = self.Y
            if Mu is None:
                Mu = self.Mu
            if Ps is None:
                Ps = self.Ps
            if Mu_minus is None:
                Mu_minus = self.Mu_minus
            if Ps_minus is None:
                Ps_minus = self.Ps_minus

            t = len(Y)
            n = len(self.A)

            # Gk：G_k = P_k A^T (P_minus_k+1)^-1
            self.Gs = np.zeros((t, n, n))
            for k in range(t):  # idx from 0 to t-1, t in total
                P_minus_inv = np.linalg.inv(Ps_minus[k])
                self.Gs[k] = Ps[k] @ self.A.T @ P_minus_inv

            # ms_k：m_s_k = m_k + G_k(m_s_k+1 - m_minus_k+1)
            self.Mu_Smoother = np.zeros((t+1, n))
            self.Mu_Smoother[-1] = Mu[-1]  # last step
            for k in reversed(range(t)):  # backwards
                self.Mu_Smoother[k] = Mu[k] + self.Gs[k] @ (self.Mu_Smoother[k+1] - Mu_minus[k])

            # Ps_k：P_s_k = P_k + G_k(P_s_k+1 - P_minus_k+1)G_k^T
            self.Ps_Smoother = np.zeros((t+1, n, n))
            self.Ps_Smoother[-1] = Ps[-1]  # last step
            for k in reversed(range(t)):
                self.Ps_Smoother[k] = Ps[k] + self.Gs[k] @ (self.Ps_Smoother[k+1] - Ps_minus[k]) @ self.Gs[k].T

            # return key values
            return {'EX Smoother': self.Mu_Smoother, 'P Smoother': self.Ps_Smoother, 'G': self.Gs}
        
        except Exception as e:
            print(e, "\n", "If Some Values Missing, Try to Call Filter() First.")

    def loglikelihood(self, theta, Y=None):
        """
        When Plot or Numerical Method
        """

        # Use build-in data
        if Y is None:
            Y = self.Y

        

        if theta is not None and self.theta == "A":
            self.kf.transition_matrices = theta
        if theta is not None and self.theta == "H":
            self.kf.observation_matrices = theta
        if theta is not None and self.theta == "Q":
            self.kf.transition_covariance = theta
        if theta is not None and self.theta == "R":
            self.kf.observation_covariance = theta
        if theta is not None and self.theta == "m0":
            self.kf.initial_state_mean = theta
        if theta is not None and self.theta == "P0":
            self.kf.initial_state_covariance = theta

        # else use build-in A for ell
        return self.kf.loglikelihood(Y)
    
    def data_for_plot_loglikelihood(self, xlim=(0, 2), num=100, Y=None):
        """
        This works only with A is dim1
        args:
            xlim: (a, b), the lower b and upper b of the function
            num: the number of discrete points
            Y: use build-in data mainly
        returns:
            As: the range for plot
            ells: (num,)    
        """
        if Y is None:
            Y = self.Y

        Thetas_for_plot = [np.array([[Theta]]) for Theta in np.linspace(xlim[0], xlim[1], num)]

        loglikelihoods = []

        for Theta in Thetas_for_plot:

            ell = self.loglikelihood(Theta, Y)

            loglikelihoods.append(ell)

        return Thetas_for_plot, loglikelihoods

class MAPParameterEstimationA(KalmanClass):
    """
    High-Level Usage: input Y and output a fitted model. Remember we have build-in data for evaluation.

    To compute the gradient of ell in closed-form, we need to use quantities from Filter part. 
    """

    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None) -> None:
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0) # use default value

    def __call__(self, *args, **kwds):
        # return super().__call__(*args, **kwds)
        pass
    
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
        model = KalmanClass(A=A, Sigma_q=self.Sigma_q, H=self.H, 
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
        metric = []

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
    
class EMParameterEstimationA(KalmanClass):
    """
    High-Level Usage: input Y and output a fitted model. Remember we have build-in data for evaluation.

    To compute the conclusions from EM Algorithm, we need to use quantities from Filter and Smoother part.
    """

    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None) -> None:
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0) # use default value

    def __call__(self, *args, **kwds):
        # return super().__call__(*args, **kwds)
        pass

    def quantities_from_Q(self, A, Y=None):
        if Y is None:
            Y = self.Y

        model = KalmanClass(A=A, Sigma_q=self.Sigma_q, H=self.H, 
                                    Sigma_r=self.Sigma_r, mu_0=self.mu_0, P_0=self.P_0)
        
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

        #
        Sigma /= T
        Phi /= T
        B /= T
        C /= T
        D /= T

        return {"Sigma": Sigma, "Phi": Phi, "B": B, "C": C, "D": D}

    def parameter_estimation(self, Y=None, num_iteration=10):

        if Y is None:
            Y = self.Y

        # init value, make sure A is > 0
        dimx = len(self.A)
        A = np.random.randn(dimx, dimx)

        print('A0', A)

        As = [A]
        metric = []

        for _ in range(num_iteration):

            result = self.quantities_from_Q(A=A, Y=Y)

            # update A = C Phi-1
            A = result["C"] @ np.linalg.inv(result["Phi"])
            m = np.linalg.norm(A-self.A, 'fro')

            As.append(A)
            metric.append(m)

        return A, As, metric

class GradientParametersEstimationAll(KalmanClass):
    def __init__(self, vars: str, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None) -> None:
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0) # use default value

        # extract theta from vars
        self.theta = vars

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

        # Function to compute mu and Sigma
        def compute_mu_sigma():
            mean = H @ A @ mu
            Sigma = H @ A @ P @ A.T @ H.T + H @ Q @ H.T + R
            return mean, Sigma

        # Compute mu and Sigma
        mean, Sigma = compute_mu_sigma()

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

        # init value
        if self.theta == "A":
            Theta = np.random.randn(*self.A.shape)
        elif self.theta == "H":
            Theta = np.random.randn(*self.H.shape)
        elif self.theta == "mu":
            Theta = np.random.randn(*self.mu_0.shape)
        elif self.theta == "P":
            Theta = np.random.randn(*self.P_0.shape)
        elif self.theta == "Q":
            Theta = np.random.randn(*self.Sigma_q.shape)
        elif self.theta == "R":
            Theta = np.random.randn(*self.Sigma_r.shape)
        
        print('Theta0:', Theta)

        Thetas = [Theta]

        for _ in range(num_iteration):
                
            print("Gradient of ell:", self.gradient_ell(Theta, Y))
            Theta = Theta + alpha * self.gradient_ell(Theta, Y)

            Thetas.append(Theta)

        return Theta, Thetas
    

if __name__ == "__main__":
    # # use build-in data

    ### EM closed form model

    # model = EMParameterEstimationA(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))

    # A, As, metric = model.parameter_estimation(num_iteration=10)
    # print(As)
    # print("A10: ", A)
    # print(metric)

    ### EM pykalman

    # model = KalmanClass(A=np.array([[1., 0.], [0., 1.]]), Sigma_q=np.array([[0.01, 0.], [0., 0.01]]), H=np.array([[1., 0.], [0., 1.]]), Sigma_r=np.array([[0.01, 0.], [0., 0.01]]), mu_0=np.array([0., 0.]), P_0=np.array([[0.01, 0.], [0., 0.01]]))
    # # #
    # kf = model.kf
    # random_A = np.random.randn(2, 2)
    # print("A0: ", random_A)
    # kf.transition_matrices = random_A

    # kf = kf.em(model.Y, n_iter=20, em_vars=['transition_matrices'])
    # print("A10: ", kf.transition_matrices)
    

    pass
