from background_subtraction import BackgroundSubtractionStrategy
from helpers import optimize_breakpoint

class SignalExtractor:
    """
    The SignalExtractor defines the interface to BackgroundSubtraction.
    """
    def __init__(self, background_strategy: BackgroundSubtractionStrategy) -> None:
        self._background_strategy = background_strategy

    def background_strategy(self) -> BackgroundSubtractionStrategy:
        """
        Just in case we need a reference to one of BackgroundSubtractionStrategy objects
        """
        return self._background_strategy

    def set_background_strategy(self, background_strategy: BackgroundSubtractionStrategy) -> None:
        """
        Usually, the SignalExtractor allows replacing a BackgroundSubtractionStrategy object at runtime.
        """
        self._background_strategy = background_strategy

    def extract_signal(self, data, planet_id):
        """The SignalExtractor delegates some work to the BackgroundSubtractionStrategy object."""

        # Find breakpoints for time steps using the optimize_breakpoint() method. We find the breakpoints inclusively in wavelength
        initial_breakpoint = (data.shape[1] // 3) - 2  # 60 --> tune this value
        transit_breakpoint = optimize_breakpoint(data[planet_id, :, :, :].mean(axis=(1,2)), initial_breakpoint)
        print('Optimized breakpoint is:', transit_breakpoint)

        # Pass this breakpoints (transition time) to the background strategy
        data_normalized, quality_metric = self._background_strategy.subtract_background(
            data, planet_id, transit_breakpoint
        )
        return data_normalized, quality_metric