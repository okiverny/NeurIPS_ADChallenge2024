from abc import ABC, abstractmethod
import numpy as np

# Strategy interface
class BackgroundSubtractionStrategy(ABC):
    @abstractmethod
    def subtract_background(self, data, planet_id, time_steps):
        pass

# Concrete strategies
class ConstantBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, time_steps):
        background = np.sum(data[planet_id, time_steps, :, :], axis=2) # as an estimator we use the sum along the spatial dimesion
        return data[planet_id] - background
    
class LinearBackgroundSubtraction(BackgroundSubtractionStrategy):
    def subtract_background(self, data, planet_id, time_steps):
        before = np.mean(data[planet_id, time_steps[:len(time_steps)//2], :, :], axis=0)
        after = np.mean(data[planet_id, time_steps[len(time_steps)//2:], :, :], axis=0)
        slope = (after - before) / (len(time_steps)//2)
        background = before + slope * np.arange(len(time_steps)).reshape(-1, 1, 1)
        return data[planet_id] - background