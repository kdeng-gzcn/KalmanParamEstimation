from Model.KalmanClass import LinearGaussianModel
import numpy as np

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def Case1():
    # Case1 x dim1 y dim1
    ## Build Model
    A = np.array([[1.]])
    H = np.array([[1.]])
    Sigma_q = np.diag([0.01])
    Sigma_r = np.diag([0.01])

    Model = LinearGaussianModel(
        A=A,
        Sigma_q=Sigma_q,
        H=H,
        Sigma_r=Sigma_r
    )

    ## Sample Data 
    mu = np.array([0])
    Sigma_p = np.diag([0.01])

    samples = Model.generate_measurement(
        mu=mu,
        Sigma_p=Sigma_p
    )

    # Plot1
    Model.plot_generated_measurement(save=True)

    ## KF Estimate based on Samples
    results = Model.kf_estimation(
        mu=mu,
        Sigma_p=Sigma_p
    )

    # Plot2 KF.plot_kf_estimation()
    Model.plot_kf_estimation(X_display=True, save=True)

    # Plot3 Plot Y and EX and real X
    Model.plot_EX_versus_X(save=True)

def Case2():
    # Case2: 2D Example with x dim2 and y dim2
    
    ## Build Model
    # State transition matrix (A)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])  # A simple motion model
    
    # Observation matrix (H)
    H = np.array([[1.1, 0.0], [0.0, 0.9]])  # Directly observe both dimensions of the state
    
    # Process noise covariance (Sigma_q)
    Sigma_q = np.diag([0.1, 0.1])  # Assume low process noise
    
    # Measurement noise covariance (Sigma_r)
    Sigma_r = np.diag([0.1, 0.1])  # Assume low measurement noise
    
    # Initialize the Kalman Filter model
    KF = LinearGaussianModel(
        A=A,
        Sigma_q=Sigma_q,
        H=H,
        Sigma_r=Sigma_r
    )
    
    ## Sample Data
    # Initial state mean (mu)
    mu = np.array([0, 0])  # Start at the origin
    
    # Initial state covariance (Sigma_p)
    Sigma_p = np.diag([0.1, 0.1])  # Assume low initial uncertainty
    
    # Generate synthetic measurements
    samples = KF.generate_measurement(
        mu=mu,
        Sigma_p=Sigma_p,
        total_timesteps=50  # Number of timesteps to generate data for
    )
    
    # Plot the generated measurements
    KF.plot_generated_measurement_scatter()
    KF.plot_generated_measurement(k=0)  # Plot for the first dimension
    KF.plot_generated_measurement(k=1)  # Plot for the second dimension
    
    ## Kalman Filter Estimation
    estimation_results = KF.kf_estimation(
        mu=mu,
        Sigma_p=Sigma_p
    )
    
    # Plot the Kalman Filter estimation
    KF.plot_kf_estimation(k=0, X_display=True)  # Plot estimation for the first dimension
    KF.plot_kf_estimation(k=1, X_display=True)  # Plot estimation for the second dimension
    
    # Compare estimated state vs true state
    KF.plot_EX_versus_X(k=0)  # For the first dimension
    KF.plot_EX_versus_X(k=1)  # For the second dimension

if __name__ == "__main__":
    # warning
    suppress_qt_warnings()

    # Case1 x dim1 y dim1
    Case1()

    # # Case2 x dimn y dimm
    # Case2()