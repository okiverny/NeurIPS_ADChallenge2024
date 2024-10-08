from abc import ABC, abstractmethod
import numpy as np

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
        egress_time_step = total_time - transit_breakpoint

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps_obscured = np.arange(ingress_time_step+2*transit_half_duration, egress_time_step-2*transit_half_duration) # avoiding edge transitions

        # Check if the constant background subtraction is valid
        quality_test = self.check_quality(estimator, time_steps_unobscured_left, time_steps_unobscured_right)

        if quality_test:
            background = estimator[time_steps_unobscured, :].mean(axis=0)
            background_err = estimator[time_steps_unobscured, :].std(axis=0)

            # Detrending the data (normalization by stellar flux)
            data_normalized = estimator/background

            #signal_plus_background = estimator[time_steps_obscured, :].mean(axis=0)
            #signal = signal_plus_background - background
            #signal_err = estimator[time_steps_obscured, :].std(axis=0) # assuming a total statistical error

            # Provide quality metric for this method as well
            quality_metric = estimator[time_steps_unobscured_left, :].mean(axis=0) / estimator[time_steps_unobscured_right, :].mean(axis=0)

            return data_normalized, quality_metric
        else:
            # Trigger an automatic fallback to a more complex method, if implemented
            return None, None, None, None, None  # Indicating this method failed
    
    def check_quality(self, estimator, time_steps_unobscured_left, time_steps_unobscured_right):
        """Check if the constant subtraction is valid by comparing mean values."""
        quality_metric = estimator[time_steps_unobscured_left, :].mean(axis=0) / estimator[time_steps_unobscured_right, :].mean(axis=0)
        return np.all(quality_metric < 1.1) and np.all(quality_metric > 0.9)
    
class LinearBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, transit_breakpoint):
        estimator = data[planet_id, :, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Setting the time points
        total_time = data.shape[1]
        transit_half_duration = total_time // 23 # time window when the stellar flux is on the half-way darkening
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))

        # implement the method here
        return None
    
    def check_quality(self, estimator, time_steps_unobscured_left, time_steps_unobscured_right):
        """Implement the quality metric as chi2/ndof of the linear fit."""
        return True
    
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