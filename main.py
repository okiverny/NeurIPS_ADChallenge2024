import numpy as np
import pandas as pd
from signal_extraction import SignalExtractor
from background_subtraction import ConstantBackgroundSubtraction, LinearBackgroundSubtraction, BackgroundSubtractionManager

if __name__ == "__main__":
    # Loading data
    data = np.load("/kaggle/input/neurips-starter/output/data_train.npy")

    planet_id = 0
    time_steps = np.arange(0, 187)  # Time steps

    # Instantiate the manager with the initial strategy (ConstantBackgroundSubtraction)
    background_manager = BackgroundSubtractionManager(ConstantBackgroundSubtraction())

    # Initialize the SignalExtractor with a constant background subtraction strategy
    extractor = SignalExtractor(ConstantBackgroundSubtraction())

    # Extract signal using the constant strategy
    signal, signal_err, background, background_err, quality_metric = extractor.extract_signal(data, planet_id)
    print('Signal Spectrum (Const):', signal)