import numpy as np
from matplotlib import pyplot as plt
import os

import sys
sys.path.append("./")

from Model.KalmanClass import GradientParameterEstimationA, EMParameterEstimationA

class FigureTemplate:
    def __init__(self, model, save_path="./Result/", save=False, title=None, xlabel=None, ylabel=None, filename=None):
        """
        A base class for setting up a standard figure template.
        
        Parameters:
        model (object): The model object containing data for plotting.
        save_path (str): The directory where figures will be saved.
        save (bool): Whether to save the figure or not.
        """
        self.model = model
        self.save_path = save_path
        self.save = save
        self.style = 'ggplot'

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.filename = filename

        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def set_up(self):
        plt.style.use(self.style)
        plt.figure(figsize=(10, 6))

    def save_figure(self):
        filepath = os.path.join(self.save_path, self.filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    def show_or_save(self):
        if self.save:
            self.save_figure()
            
        else:
            plt.show()

    def title_labels(self):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()

class PlotMeasurement(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="Generated Measurement", xlabel="Timestep", ylabel="Value", filename="Generated Measurement.png", dim=0):
        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        # Plot X
        plt.plot(
            range(self.model.X.shape[1]),
            self.model.X[dim, :],
            color='b',
            label=f"x[{dim}]"
            )
        
        # Plot Y
        plt.scatter(range(1, self.model.X.shape[1]), self.model.Y[dim, :], label=f"y[{dim}]")

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

class PlotKFAndMeasurement(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="KF Estimation", xlabel="Timestep", ylabel="Value", filename="KF Estimation.png", dim=0):
        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        self.model.Filter()

        # Plot X
        plt.plot(
            range(self.model.X.shape[1]),
            self.model.X[dim, :],
            color='b',
            label=f"x[{dim}]"
            )
        
        # Plot Mus
        plt.plot(
            range(self.model.Mu.shape[1]), 
            self.model.Mu[dim, :],
            linestyle="--",
            color='c',
            label=f"mu[{dim}]"
            )
        
        # Plot Y
        plt.scatter(range(1, self.model.X.shape[1]), self.model.Y[dim, :], label=f"y[{dim}]")

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

class PlotFilterSmootherMeasurement(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="Kalman Filter and Smoother", xlabel="Timestep", ylabel="Value", filename="Kalman Filter and Smoother.png", dim=0):
        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        self.model.Filter()
        self.model.Smoother()

        # Plot X
        plt.plot(
            range(self.model.X.shape[0]),
            self.model.X[:, dim],
            color='b',
            label=f"x[{dim}]"
            )
        
        # Plot Mus (t+1) * n
        plt.plot(
            range(self.model.Mu.shape[0]), 
            self.model.Mu[:, dim],
            linestyle="--",
            color='c',
            label=f"mu[{dim}]"
            )
        
        # Plot Mu_Smoother
        plt.plot(
            range(self.model.Mu_Smoother.shape[0]), 
            self.model.Mu_Smoother[:, dim],
            linestyle="--",
            color='m',
            label=f"mu Smoother[{dim}]"
            )
        
        # Plot Y
        plt.scatter(range(1, self.model.Y.shape[0]+1), self.model.Y[:, dim], label=f"y[{dim}]")

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

class PlotLoglikelihood(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="Loglikelihood", xlabel="Theta", ylabel=r"$\ell(\theta \mid Y)$", filename="Loglikelihood.png"):

        assert isinstance(model, GradientParameterEstimationA), "Not MAP Parameter Estimation"

        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        As_for_plot, ells = self.model.data_for_plot_loglikelihood()

        plt.plot(np.array(As_for_plot).squeeze(), ells, label=r"$\ell(A \mid Y)$")
        plt.axvline(x=self.model.A.squeeze(), color='b', linestyle='--', label=f'True Value {self.model.A.squeeze()}')

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

class PlotMAPWithLoglikelihood(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="MAP Param Estimation", xlabel="Theta", ylabel=r"$\ell(\theta \mid Y)$", filename="MAP Iteratin and Loglikelihood.png"):

        assert isinstance(model, GradientParameterEstimationA), "Not MAP Parameter Estimation"

        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        _, As, _ = self.model.parameter_estimation(alpha=0.001, numerical=False)

        As_for_plot, ells = self.model.data_for_plot_loglikelihood()

        plt.plot(np.array(As_for_plot).squeeze(), ells, label=r"$\ell(A \mid Y)$")
        plt.axvline(x=self.model.A.squeeze(), color='b', linestyle='--', label=f'True Value {self.model.A.squeeze()}')
        for idx, A in enumerate(As):
            plt.axvline(x=A.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(A.squeeze(), 3)}')

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

class PlotEMWithLoglikelihood(FigureTemplate):
    def __init__(self, model, save_path="./Result/", save=False, title="EM Param Estimation", xlabel="Theta", ylabel=r"$\ell(\theta \mid Y)$", filename="EM Iteratin and Loglikelihood.png"):

        assert isinstance(model, EMParameterEstimationA), "Not EM Parameter Estimation"

        super().__init__(model, save_path, save, title, xlabel, ylabel, filename)

        # Plot
        self.set_up()

        ##### Core Part

        _, As, _ = self.model.parameter_estimation()

        As_for_plot, ells = self.model.data_for_plot_loglikelihood()

        plt.plot(np.array(As_for_plot).squeeze(), ells, label=r"$\ell(A \mid Y)$")
        plt.axvline(x=self.model.A.squeeze(), color='b', linestyle='--', label=f'True Value {self.model.A.squeeze()}')
        for idx, A in enumerate(As):
            plt.axvline(x=A.squeeze(), color='c', linestyle=':', label=f'Theta{idx}: {np.round(A.squeeze(), 3)}')

        #####
        
        self.title_labels()
        
        # Call the method to either show or save the plot
        self.show_or_save()

if __name__ == "__main__":

    import sys
    sys.path.append('./')

    from Model.KalmanClass import KalmanClass

    ### Task1

    # model = KalmanClass()

    # PlotMeasurement(model=model)
    # PlotMeasurement(model=model, save=True, save_path="./Result/", filename="demo.png")

    # PlotKFAndMeasurement(model=model)
    # PlotKFAndMeasurement(model=model, save=True, filename="demo.png")

    # PlotFilterSmootherMeasurement(model=model, save=True)

    ### Task2

    model = GradientParameterEstimationA()

    PlotMAPWithLoglikelihood(model=model)
    # PlotMAPWithLoglikelihood(model=model, save=True)

    # model = EMParameterEstimationA()
    # PlotEMWithLoglikelihood(model=model, save=False)

    ### Task3


