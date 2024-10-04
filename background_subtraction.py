from abc import ABC, abstractmethod
import numpy as np

# Strategy interface
class BackgroundSubtractionStrategy(ABC):
    """This is the interface that declares operations common to all background subtraction algorithms/methods."""
    @abstractmethod
    def subtract_background(self, data, planet_id, time_steps):
        pass

    @abstractmethod
    def check_quality(self, data, planet_id, time_steps):
        """Check the quality of the background subtraction method."""
        pass

# Concrete Strategies
class ConstantBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, time_steps):
        estimator = data[planet_id, time_steps, :, :].sum(axis=2)   # Sum along spatial dimension, it contains signal+background.

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, 56)
        time_steps_unobscured_right = np.arange(130, 187)
        time_steps_unobscured = np.concatenate((time_steps_unobscured_left, time_steps_unobscured_right))
        time_steps_obscured = np.arange(70, 116) # avoiding edge transitions

        # Check if the constant background subtraction is valid
        quality_test = self.check_quality(estimator, time_steps_unobscured_left, time_steps_unobscured_right)

        if quality_test:
            background = estimator[time_steps_unobscured, :].mean(axis=0)
            background_err = estimator[time_steps_unobscured, :].std(axis=0)

            signal_plus_background = estimator[time_steps_obscured, :].mean(axis=0)
            signal = signal_plus_background - background
            signal_err = estimator[time_steps_obscured, :].std(axis=0) # assuming a total statistical error

            return signal, signal_err, background, background_err
        else:
            # Trigger an automatic fallback to a more complex method
            return None, None, None, None  # Indicating this method failed
    
    def check_quality(self, estimator, time_steps_unobscured_left, time_steps_unobscured_right):
        """Check if the constant subtraction is valid by comparing mean values."""
        quality_metric = estimator[time_steps_unobscured_left, :].mean(axis=0) / estimator[time_steps_unobscured_right, :].mean(axis=0)
        return np.all(quality_metric < 1.1) and np.all(quality_metric > 0.9)
    
class LinearBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, time_steps):
        before = np.mean(data[planet_id, time_steps[:len(time_steps)//2], :, :], axis=0)
        after = np.mean(data[planet_id, time_steps[len(time_steps)//2:], :, :], axis=0)
        slope = (after - before) / (len(time_steps)//2)
        background = before + slope * np.arange(len(time_steps)).reshape(-1, 1, 1)
        return data[planet_id] - background
    
    def check_quality(self, estimator, time_steps_unobscured_left, time_steps_unobscured_right):
        """Linear subtraction does not require a quality check."""
        return True
    
class BackgroundSubtractionManager:
    """Manager Class to Control Strategy and Fallback"""
    def __init__(self, initial_strategy):
        self._background_strategy = initial_strategy
    
    def subtract_background(self, data, planet_id, time_steps):
        signal, background, background_err, signal_err = self._background_strategy.subtract_background(data, planet_id, time_steps)
        
        # If ConstantBackgroundSubtraction fails, fallback to LinearBackgroundSubtraction
        if signal is None:
            print("Switching to Linear Background Subtraction...")
            self._background_strategy = LinearBackgroundSubtraction()
            signal, background, background_err, signal_err = self._background_strategy.subtract_background(data, planet_id, time_steps)
        
        return signal, background, background_err, signal_err