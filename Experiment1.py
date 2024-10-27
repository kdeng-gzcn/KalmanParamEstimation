from Model.KalmanClass import KalmanClass
import numpy as np

# This is for kalman filter, try some params combination for intuition.
# dim = 1 for plotting

# for a stable model, try different sigma for dynamic and measurement model.
# to see the model efficiency.

# we need a metric to evaluate the model efficiency, e.g. MSE

def load_model(A = np.array([[1.]]), H = np.array([[1.]]), Sigma_q = np.diag([0.01]), Sigma_r = np.diag([0.01])):

    Model = KalmanClass(
        A=A,
        Sigma_q=Sigma_q,
        H=H,
        Sigma_r=Sigma_r
    )

    return Model

def generate_data_and_kf_filter(model):

    ## Sample Data 
    mu = np.array([0])
    Sigma_p = np.diag([0.01])

    samples = model.generate_measurement(
        mu=mu,
        Sigma_p=Sigma_p
    )

    results = model.kf_estimation(
        mu=mu,
        Sigma_p=Sigma_p
    )

    return samples['Xtn'], results['EXtn']

def MSE(EX, X):

    Y = np.array(EX)
    Target = np.array(X)

    return np.sum((Y - Target) ** 2) / len(Y)

def experiment(Sigma_q, Sigma_r):

    mses = []

    for _ in range(5):
        model = load_model(Sigma_q=Sigma_q, Sigma_r=Sigma_r)
        X, EX = generate_data_and_kf_filter(model=model)
        mse = MSE(EX=EX, X=X)
        mses.append(mse)

    return sum(mses) / len(mses)

def main(Sigma_qs, Sigma_rs):
    '''
    args:
        Sigma_qs: list of (dim_x, )
        Sigma_rs: list of (dim_y, )
    return:
        Mat of results: (dim_y, dim_x)
    '''
    
    MSEs = []

    for Sigma_q in Sigma_qs:

        

        for Sigma_r in Sigma_rs:

            mse = experiment(Sigma_q, Sigma_r)

            MSEs.append(mse)

    
    MSEs = np.array(MSEs).reshape((len(Sigma_qs), len(Sigma_rs)))

    # print(MSEs)

    return MSEs

if __name__ == "__main__":

    Sigma_qs = [np.diag([0.01]), np.diag([0.02]), np.diag([0.05]), np.diag([0.1]), np.diag([0.2]), np.diag([0.5])]
    Sigma_rs = [np.diag([0.01]), np.diag([0.02]), np.diag([0.05]), np.diag([0.1]), np.diag([0.2]), np.diag([0.5])]
    
    MSEs = main(Sigma_qs=Sigma_qs, Sigma_rs=Sigma_rs)

    import pandas as pd

    df = pd.DataFrame(MSEs, index=[f'{Sigma_q.squeeze()}' for Sigma_q in Sigma_qs], columns=[f'{Sigma_r.squeeze()}' for Sigma_r in Sigma_rs])

    print(df)
