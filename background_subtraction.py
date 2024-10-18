from abc import ABC, abstractmethod
import numpy as np
import scipy

from helpers import linear_func, quadratic_func, pol3_func, calc_chisquare

# Strategy interface
class BackgroundSubtractionStrategy(ABC):
    """This is the interface that declares operations common to all background subtraction algorithms/methods."""
    @abstractmethod
    def subtract_background(self, data, planet_id):
        pass

    @abstractmethod
    def check_quality(self, data, planet_id, time_steps):
        """Check the quality of the background subtraction method."""
        pass

# Concrete Strategies
class ConstantBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, transit_breakpoint):
        estimator = data[planet_id, :, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Setting the time points
        total_time = data.shape[1]
        transit_half_duration = total_time // 23 # time window when the stellar flux is on the half-way darkening
        midpoint = total_time // 2  # planet is at the middle
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint + transit_half_duration

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps_obscured = np.arange(ingress_time_step+2*transit_half_duration, egress_time_step-2*transit_half_duration) # avoiding edge transitions

        # Check if the constant background subtraction is valid
        quality_metric = estimator[time_steps_unobscured_left, :].mean(axis=0) / estimator[time_steps_unobscured_right, :].mean(axis=0)
        quality_test = self.check_quality(quality_metric)

        if quality_test:
            background = estimator[time_steps_unobscured, :].mean(axis=0)
            background_err = estimator[time_steps_unobscured, :].std(axis=0)

            # Detrending the data (normalization by stellar flux)
            data_normalized = estimator/background

            # Provide quality metric for this method as well
            quality_metric = estimator[time_steps_unobscured_left, :].mean(axis=0) / estimator[time_steps_unobscured_right, :].mean(axis=0)

            return data_normalized, quality_metric
        else:
            # Trigger an automatic fallback to a more complex method, if implemented
            return None, None  # Indicating this method failed
    
    def check_quality(self, quality_metric):
        """Check if the constant subtraction is valid by comparing mean values."""
        return np.all(quality_metric < 1.1) and np.all(quality_metric > 0.9)
    
class LinearBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, transit_breakpoint):
        estimator = data[planet_id, :, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Setting the time points
        total_time = data.shape[1]
        transit_half_duration = total_time // 23 # = 8 ; time window when the stellar flux is on the half-way darkening
        midpoint = total_time // 2  # planet is at the middle
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint + transit_half_duration

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps = np.arange(0, total_time)

        # Get the corresponding values
        y_left = estimator[time_steps_unobscured_left, :]
        y_right = estimator[time_steps_unobscured_right, :]
        y = estimator[time_steps_unobscured, :]
        y_err = np.ones_like(y) * np.minimum(y_left.std(axis=0), y_right.std(axis=0))

        # Initialize the lists for important quantities
        par0, par1, quality_metric = [], [], []
        par0_err, par1_err = [], []
        data_normalized = []

        # Loop over wavelengths
        for lambda_i in range(y.shape[1]):
            popt, pcov = scipy.optimize.curve_fit(linear_func, time_steps_unobscured, y[:,lambda_i], p0=np.array([0, 3.83e+08]), maxfev=800)
            popt_err = np.sqrt(np.diag(pcov))

            # Append fit results
            par0.append(popt[0])
            par1.append(popt[1])
            par0_err.append(popt_err[0])
            par1_err.append(popt_err[1])

            # Compute and append the fit quality metric chi2/NDF
            chi2 = calc_chisquare(y[:, lambda_i], y_err[:, lambda_i], linear_func(time_steps_unobscured, *popt))
            NDF = len(y) - len(popt)
            quality_metric.append(chi2/NDF)

            # Detrending the data (normalization by stellar flux)
            data_normalized.append( estimator[:,lambda_i] / linear_func(time_steps, *popt) )

        # implement the method here
        return np.array(data_normalized).T, np.array(quality_metric)
    
    def check_quality(self, quality_metric):
        """Quality metric as chi2/ndof of the linear fit."""
        return np.all(quality_metric < 1000.0)
    

class QuadraticBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, transit_breakpoint):
        estimator = data[planet_id, :, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Setting the time points
        total_time = data.shape[1]
        transit_half_duration = total_time // 23 #  = 8 ; time window when the stellar flux is on the half-way darkening
        midpoint = total_time // 2  # planet is at the middle
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint + transit_half_duration

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps = np.arange(0, total_time)

        # Get the corresponding values
        y_left = estimator[time_steps_unobscured_left, :]
        y_right = estimator[time_steps_unobscured_right, :]
        y = estimator[time_steps_unobscured, :]
        y_err = np.ones_like(y) * np.minimum(y_left.std(axis=0), y_right.std(axis=0))

        # Initialize the lists for important quantities
        par0, par1 = [], []
        par0_err, par1_err = [], []
        quality_metric = []
        data_normalized = []

        # Loop over wavelengths
        for lambda_i in range(y.shape[1]):
            popt, pcov = scipy.optimize.curve_fit(quadratic_func, time_steps_unobscured, y[:,lambda_i], p0=np.array([50, -20000, 3.83e+08]), maxfev=800)
            popt_err = np.sqrt(np.diag(pcov))

            # Append fit results
            par0.append(popt[0])
            par1.append(popt[1])
            par0_err.append(popt_err[0])
            par1_err.append(popt_err[1])

            # Compute and append the fit quality metric chi2/NDF
            chi2 = calc_chisquare(y[:, lambda_i], y_err[:, lambda_i], quadratic_func(time_steps_unobscured, *popt))
            NDF = len(y) - len(popt)
            quality_metric.append(chi2/NDF)

            # Detrending the data (normalization by stellar flux)
            data_normalized.append( estimator[:,lambda_i] / quadratic_func(time_steps, *popt) )

        # implement the method here
        return np.array(data_normalized).T, np.array(quality_metric)
    
    def check_quality(self, quality_metric):
        """Quality metric as chi2/ndof of the linear fit."""
        return np.all(quality_metric < 1000.0)
    
class CubicBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, transit_breakpoint, lambda_step=1):
        estimator = data[planet_id, :, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Setting the time points
        total_time = data.shape[1]
        transit_half_duration = total_time // 23 #  = 8 ; time window when the stellar flux is on the half-way darkening
        midpoint = total_time // 2  # planet is at the middle
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint + transit_half_duration

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps = np.arange(0, total_time)

        # Get the corresponding values
        y_left = estimator[time_steps_unobscured_left, :]
        y_right = estimator[time_steps_unobscured_right, :]
        y = estimator[time_steps_unobscured, :]
        y_err = np.ones_like(y) * np.minimum(y_left.std(axis=0), y_right.std(axis=0))

        # Initialize the lists for important quantities
        par0, par1 = [], []
        par0_err, par1_err = [], []
        quality_metric = []
        data_normalized = []

        # Loop over wavelengths
        for lambda_i in range(y.shape[1]):
            popt, pcov = scipy.optimize.curve_fit(pol3_func, time_steps_unobscured, y[:,lambda_i], p0=np.array([0, 50, -20000, 3.83e+08]), maxfev=800)
            popt_err = np.sqrt(np.diag(pcov))

            # Append fit results
            par0.append(popt[0])
            par1.append(popt[1])
            par0_err.append(popt_err[0])
            par1_err.append(popt_err[1])

            # Compute and append the fit quality metric chi2/NDF
            chi2 = calc_chisquare(y[:, lambda_i], y_err[:, lambda_i], pol3_func(time_steps_unobscured, *popt))
            NDF = len(y) - len(popt)
            quality_metric.append(chi2/NDF)

            # Detrending the data (normalization by stellar flux)
            data_normalized.append( estimator[:,lambda_i] / pol3_func(time_steps, *popt) )

        # implement the method here
        return np.array(data_normalized).T, np.array(quality_metric)

    def check_quality(self, quality_metric):
        """Quality metric as chi2/ndof of the linear fit."""
        return np.all(quality_metric < 1000.0)

# The manager not yet used in the implementation!
class BackgroundSubtractionManager:
    """Manager Class to Control Strategy and Fallback"""
    def __init__(self, initial_strategy):
        self._background_strategy = initial_strategy
    
    def subtract_background(self, data, planet_id):
        signal, background, background_err, signal_err = self._background_strategy.subtract_background(data, planet_id)
        
        # If ConstantBackgroundSubtraction fails, fallback to LinearBackgroundSubtraction
        if signal is None:
            print("Switching to Linear Background Subtraction...")
            self._background_strategy = LinearBackgroundSubtraction()
            signal, background, background_err, signal_err = self._background_strategy.subtract_background(data, planet_id)
        
        return signal, background, background_err, signal_err