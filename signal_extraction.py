from background_subtraction import BackgroundSubtractionStrategy

class SignalExtractor:
    def __init__(self, background_strategy: BackgroundSubtractionStrategy) -> None:
        self.background_strategy = background_strategy

    def set_background_strategy(self, background_strategy: BackgroundSubtractionStrategy) -> None:
        self.background_strategy = background_strategy

    def extract_signal(self, data, planet_id, time_steps):
        background_subtracted_data = self.background_strategy.subtract_background(data, planet_id, time_steps)
        return background_subtracted_data