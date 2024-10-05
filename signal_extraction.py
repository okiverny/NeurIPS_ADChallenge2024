from background_subtraction import BackgroundSubtractionStrategy

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
        """
        The SignalExtractor delegates some work to the BackgroundSubtractionStrategy object.
        """
        signal, signal_err, background, background_err, quality_metric = self._background_strategy.subtract_background(data, planet_id)
        return signal, signal_err, background, background_err, quality_metric