import numpy as np
import pandas as pd
from signal_extraction import SignalExtractor
from background_subtraction import ConstantBackgroundSubtraction, LinearBackgroundSubtraction

if __name__ == "__main__":
    # Loading data
    data = np.load("/kaggle/input/neurips-starter/output/data_train.npy")

    planet_id = 0
    time_steps = np.arange(55, 130)  # Time steps when planet is in front of the star

    # Initialize the SignalExtractor with a constant background subtraction strategy
    extractor = SignalExtractor(ConstantBackgroundSubtraction())

    # Extract signal using the constant strategy
    signal_spectrum_const = extractor.extract_signal(data, planet_id, time_steps)
    print('Signal Spectrum (Const):', signal_spectrum_const)