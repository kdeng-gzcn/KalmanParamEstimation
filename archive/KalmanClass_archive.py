import numpy as np
from matplotlib import pyplot as plt

from filterpy.kalman import KalmanFilter

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

class LinearGaussianModel:
    def __init__(self, A, Sigma_q, H, Sigma_r) -> None:
        # Theta
        self.A = A
        self.Sigma_q = Sigma_q
        self.H = H
        self.Sigma_r = Sigma_r

        # KF Object
        self.kf = KalmanFilter(dim_x=len(A), dim_z=len(H))
        # Dynamic
        self.kf.F = A
        self.kf.Q = Sigma_q

        # Measurement
        self.kf.H = H
        self.kf.R = Sigma_r

    def dynamic(self, x_k):
        q_k = np.random.multivariate_normal(mean=np.zeros(len(x_k)), cov=self.Sigma_q)
        x_k_plus_1 = np.matmul(self.A, x_k) + q_k
        return x_k_plus_1

    def measurement(self, x_k):
        r_k = np.random.multivariate_normal(mean=np.zeros(len(x_k)), cov=self.Sigma_r)
        y_k = np.matmul(self.H, x_k) + r_k
        return y_k
    
    def generate_measurement(self, mu, Sigma_p, total_timesteps=50):
        """
        mu: mu_0
        P: P_0
        """
        x_0 = np.random.multivariate_normal(mean=mu, cov=Sigma_p)

        self.X = [x_0] # t*n
        self.Y = [] # t*m

        ### for loop
        x_k = x_0
        for _ in range(total_timesteps):
            x_k = self.dynamic(x_k)
            y_k = self.measurement(x_k)
            self.X.append(x_k)
            self.Y.append(y_k)

        return {'Xtn': self.X, 'Ytm': self.Y}
    
    def kf_estimation(self, mu, Sigma_p, Y=None):

        if Y is None:
            Y = self.Y

        # Initial Value
        self.kf.x = mu # EX = mu
        self.kf.P = Sigma_p

        # Note that Y is measurement matrix
        self.EX = [mu]

        # forloop t times
        for y in Y:
            # update mu and P for x
            self.kf.predict() 
            # update mu based on y in Y
            self.kf.update(y)
            # get mu|y
            self.EX.append(self.kf.x)

        return {'EXtn': self.EX, 'Ytn': Y}
        
    def plot_generated_measurement(self, k=0):
        plt.style.use('ggplot')  # Apply ggplot style

        plt.figure(figsize=(10, 6))
        # Plot X
        plt.plot(
            range(len(self.X[1:])), 
            [x[k] for x in self.X[1:]]
            )
        # Plot Y
        plt.scatter(range(len(self.Y)), [y[k] for y in self.Y])

        plt.title("Measurement Generation: Value v.s. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend([f"x[{k}]", f"y[{k}]"])
        plt.show()

    def plot_generated_measurement_scatter(self):
        try:
            plt.style.use('ggplot')  # Apply ggplot style

            plt.figure(figsize=(10, 6))
            plt.scatter([x[0] for x in self.X[1:]], [x[1] for x in self.X[1:]])
            plt.scatter([y[0] for y in self.Y], [y[1] for y in self.Y])
            plt.title("Measurement Generation: Dim2 v.s. Dim1")
            plt.xlabel("Dim1")
            plt.ylabel("Dim2")
            plt.legend(["xt", "yt"])
            plt.show()
        except Exception as e:
            print(e)

    def plot_kf_estimation(self, k=0, X_display=False):
        plt.style.use('ggplot')  # Apply ggplot style

        plt.figure(figsize=(10, 6))
        # Plot X
        if X_display:
            plt.plot(range(len(self.X[1:])), [x[k] for x in self.X[1:]])
        # Plot EX
        plt.plot(
            range(len(self.EX[1:])), 
            [Ex[k] for Ex in self.EX[1:]],
            linestyle="--"
            )
        # Plot Y
        plt.scatter(range(len(self.Y)), [y[k] for y in self.Y])
        plt.title("KF Estimation: Value v.s. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        if X_display:
            plt.legend([f"x[{k}]", f"Ex[{k}]", f"y[{k}]"])
        else:
            plt.legend([f"Ex[{k}]", f"y[{k}]"])
        plt.show()

    def plot_EX_versus_X(self, k=0):
        plt.style.use('ggplot')  # Apply ggplot style

        plt.figure(figsize=(10, 6))
        # Plot EX v.s. X
        plt.scatter(
            [x[k] for x in self.X[1:]], 
            [Ex[k] for Ex in self.EX[1:]],
            marker="^",
            color="c"
            )
        # Plot y=x
        plt.plot(
            np.linspace(start=min([x[k] for x in self.X[1:]]), stop=max([Ex[k] for Ex in self.EX[1:]]), num=50), 
            np.linspace(start=min([x[k] for x in self.X[1:]]), stop=max([Ex[k] for Ex in self.EX[1:]]), num=50)
            )
        plt.title(f"KF Estimation: EX[{k}] v.s. X[{k}]")
        plt.xlabel("X")
        plt.ylabel("EX")
        plt.legend(["Estimation", "Baseline"])
        plt.show()

    def plot_generated_measurement_scatter(self, save=False):
        try:
            plt.style.use('ggplot')  # Apply ggplot style

            plt.figure(figsize=(10, 6))
            plt.scatter([x[0] for x in self.X[1:]], [x[1] for x in self.X[1:]])
            plt.scatter([y[0] for y in self.Y], [y[1] for y in self.Y])
            title = "Measurement Generation: Dim2 v.s. Dim1"
            plt.title(title)
            plt.xlabel("Dim1")
            plt.ylabel("Dim2")
            plt.legend(["xt", "yt"])

            if save:
                plt.savefig('Measurement Generation.png', dpi=300, bbox_inches='tight')

            plt.show()
        except Exception as e:
            print(e)

    def plot_EX_versus_X(self, k=0, save=False):
        plt.style.use('ggplot')  # Apply ggplot style

        plt.figure(figsize=(10, 6))
        # Plot EX v.s. X
        plt.scatter(
            [x[k] for x in self.X[1:]], 
            [Ex[k] for Ex in self.EX[1:]],
            marker="^",
            color="black"
            )
        # Plot y=x
        plt.plot(
            np.linspace(start=min([x[k] for x in self.X[1:]]), stop=max([Ex[k] for Ex in self.EX[1:]]), num=50), 
            np.linspace(start=min([x[k] for x in self.X[1:]]), stop=max([Ex[k] for Ex in self.EX[1:]]), num=50)
            )
        title = f"KF Estimation: EX[{k}] v.s. X[{k}]"
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("EX")
        plt.legend(["Estimation", "Baseline"])

        if save:
                plt.savefig('KF Estimation EX X.png', dpi=300, bbox_inches='tight')

        plt.show()        


class ParameterEstimation:

    def __init__(self) -> None:
        pass

