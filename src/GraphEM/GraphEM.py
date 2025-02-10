# import class and functions
from Model.KalmanClass import KalmanClass
from Model import funcs_GraphEM
# import pkg
import numpy as np

class GraphEMforA(KalmanClass):

    def __init__(self, A=None, Sigma_q=None, H=None, Sigma_r=None, mu_0=None, P_0=None, reg_name="Laplace"):

        """

        Generate the build-in data

        """

        # set up true model params and get X, Y
        super().__init__(A, Sigma_q, H, Sigma_r, mu_0, P_0)

        self.reg_name = reg_name

    def quantities_from_Q(self, Theta, Y=None):

        if Y is None:
            Y = self.Y

        # load model with default value but soon would update
        model = KalmanClass(A=self.A, Sigma_q=self.Sigma_q, H=self.H, 
                                    Sigma_r=self.Sigma_r, mu_0=self.mu_0, P_0=self.P_0)
        
        # update model with current iteration theta
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
        
        model.Filter(Y=Y) # make sure the quantities are from build-in data
        """
        
        return {'EX Smoother': self.Mu_Smoother, 'P Smoother': self.Ps_Smoother, 'G': self.Gs}
        
        """
        smoother_dict = model.Smoother(Y=Y) # make sure the quantities are from build-in data

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

        # final answer
        Sigma /= T
        Phi /= T
        B /= T
        C /= T
        D /= T

        results = {
            "Sigma": Sigma, 
            "Phi": Phi, 
            "B": B, 
            "C": C, 
            "D": D, 
            "EX Smoother": Mu_Smoother, 
            "P Smoother": Ps_Smoother,
        }

        return results
    
    def Douglas_Rachford(self, A=None, gamma=None, Sigma=None, Phi=None, C=None, T=None, Q=None, xi=None):

        """

        The core process from MATLAB with functions in funcs_GraphEM.py

        This is optimization method for dealing with some object function hard to optim directly

        We dont need to explicitly compute Q(A, Ai)

        We simplify the question into computing PhiBCD... and do optim process directly

        """

        # unpakck the params and set up hyper params
        gamma = gamma
        Sigma = Sigma
        Phi = Phi
        C = C
        T = T
        Q = Q
        xi = xi

        num_iteration = 1000
        
        # set up start point, which is A from last step, to be start hidden var
        Y = A
        opt_A = A
        obj_q = funcs_GraphEM.q_wrt_A(Q=self.Sigma_q, A=opt_A, Sigma=Sigma, Phi=Phi, C=C, T=T)
        if self.reg_name == "Laplace":
            obj_norm = funcs_GraphEM.L1_wrt_A(A=opt_A, gamma=gamma)
        if self.reg_name == "Gaussian":
            obj_norm = funcs_GraphEM.Gaussian_Prior_wrt_A(A=opt_A, gamma=gamma)
        if self.reg_name == "Laplace_Gaussian":
            obj_norm = funcs_GraphEM.L1_Gaussian_Prior_wrt_A(A=opt_A, gamma=gamma)
        obj_list_em = [obj_q + obj_norm] # Q = q + reg

        # start iteration
        for idx_iteration in range(num_iteration): # add new stop condition

            # update optim var
            opt_A = funcs_GraphEM.opt_wrt_L1(A=Y, gamma=gamma)
            if self.reg_name == "Laplace":
                opt_A = funcs_GraphEM.opt_wrt_L1(A=Y, gamma=gamma)
            if self.reg_name == "Gaussian":
                opt_A = funcs_GraphEM.opt_wrt_Gaussian_Prior(A=Y, gamma=gamma)
            if self.reg_name == "Laplace_Gaussian":
                opt_A = funcs_GraphEM.opt_wrt_L1_Gaussian_Prior(A=Y, gamma=gamma)

            # print("A from L1 opt:\n", opt_A)

            # store obj in each step
            obj_q = funcs_GraphEM.q_wrt_A(Q=self.Sigma_q, A=opt_A, Sigma=Sigma, Phi=Phi, C=C, T=T)
            if self.reg_name == "Laplace":
                obj_norm = funcs_GraphEM.L1_wrt_A(A=opt_A, gamma=gamma)
            if self.reg_name == "Gaussian":
                obj_norm = funcs_GraphEM.Gaussian_Prior_wrt_A(A=opt_A, gamma=gamma)
            if self.reg_name == "Laplace_Gaussian":
                obj_norm = funcs_GraphEM.L1_Gaussian_Prior_wrt_A(A=opt_A, gamma=gamma)
            obj_list_em.append(obj_q + obj_norm)

            # compute optim for another part in object function
            # note that here we use 2 * A - Y
            V = funcs_GraphEM.opt_wrt_q(A=2 * opt_A - Y, C=C, Phi=Phi, Q=Q, T=T)

            # print("A from q opt:\n", V)

            # update hidden var
            Y = Y + V - opt_A

            # check stop condition
            # if consecutive values are very similar, i.e. less than eps
            if idx_iteration > 0 and np.abs(obj_list_em[idx_iteration] - obj_list_em[idx_iteration - 1]) <= xi:
                print(f"Douglas Rachford converged after iteration {idx_iteration+1}")
                break
            # if we are actually optimizing, i.e. strictly decreasing 
            # no need, its decreasing

        return opt_A

    def parameter_estimation(self, Y=None, num_iteration=100, gamma=0.001, eps=1e-5, xi=1e-5):

        print(f"GraphEM with {self.reg_name}")

        if Y is None:
            Y = self.Y # use build-in data if not assigned values to Y

        """
        
        init A
        
        """

        self.theta = "A" # make sure to use self.loglikelihood function

        init_A = np.zeros_like(self.A)

        for i in range(len(init_A)):
            for j in range(len(init_A)):
                init_A[i, j] = 0.1 ** abs(i - j)

        U, S, VT = np.linalg.svd(init_A)
        max_singular_value = np.max(S)
        coef = 0.99 / max_singular_value
        init_A = coef * init_A

        fnorm = np.linalg.norm(init_A - self.A, 'fro')
        loglikelihood = self.loglikelihood(theta=init_A, Y=Y)

        # print('A0:\n', init_A)
        # print("F-norm(A0, trueA)0:\n", fnorm)
        # print("Loglikelihood(A0)0:\n", -loglikelihood)

        # model params
        A = init_A

        # set up hyperparams
        T = len(Y)
        num_iteration = num_iteration
        gamma = gamma
        eps = eps
        xi = xi

        # init our process log
        # use loglikelihood for A | Y as real obj func
        A_list = [A]
        Fnorm_list = [fnorm]
        loglikelihood_list = [-loglikelihood]
        other_metric_list = []
        obj_q_list = []
        obj_norm_list = []
        obj_list = []

        for idx_iteration in range(num_iteration):
            
            # return {"Sigma": Sigma, "Phi": Phi, "B": B, "C": C, "D": D, "EX Smoother": Mu_Smoother, "P Smoother": Ps_Smoother}
            result = self.quantities_from_Q(Theta=A, Y=Y)

            # unpack the result
            Sigma = result['Sigma']
            Phi = result['Phi']
            B = result["B"]
            C = result['C']
            D = result['D']

            # object func for last step
            obj_q = funcs_GraphEM.q_wrt_A(Q=self.Sigma_q, A=A, Sigma=Sigma, Phi=Phi, C=C, T=T)
            obj_norm = funcs_GraphEM.L1_wrt_A(A=A, gamma=gamma)
            obj_q_list.append(obj_q)
            obj_norm_list.append(obj_norm)
            obj_list.append(obj_q + obj_norm)

            # optim em
            A = self.Douglas_Rachford(A=A, gamma=gamma, Sigma=Sigma, Phi=Phi, C=C, T=T, Q=self.Sigma_q, xi=xi)
            fnorm = np.linalg.norm(A - self.A, 'fro')
            loglikelihood = self.loglikelihood(theta=A, Y=Y)

            # object func for this step
            obj_q = funcs_GraphEM.q_wrt_A(Q=self.Sigma_q, A=A, Sigma=Sigma, Phi=Phi, C=C, T=T)
            obj_norm = funcs_GraphEM.L1_wrt_A(A=A, gamma=gamma)

            # store answers with list
            A_list.append(A)
            Fnorm_list.append(fnorm)
            loglikelihood_list.append(-loglikelihood)

            # check stop condition
            # if consecutive values are very similar, i.e. less than eps
            # try to use loglikelihood as stop condition
            if idx_iteration > 0 and np.abs(obj_list[idx_iteration] - obj_list[idx_iteration - 1]) <= eps:
                print(f"EM converged after iteration {idx_iteration+1}")
                break
            # if we are actually optimizing, i.e. strictly decreasing
            # use func.Q, probably no need
            

        obj_q_list.append(obj_q)
        obj_norm_list.append(obj_norm)
        obj_list.append(obj_q + obj_norm)

        results = {
            "A iterations": A_list, 
            "Fnorm iterations": Fnorm_list, 
            "Simple Q iterations": obj_list, 
            "General Q iteratioins": None, 
            "Loglikelihood iterations": loglikelihood_list,
        }

        return results
