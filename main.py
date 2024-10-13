import numpy as np
import pandas as pd
from signal_extraction import SignalExtractor
from background_subtraction import ConstantBackgroundSubtraction, LinearBackgroundSubtraction, QuadraticBackgroundSubtraction, BackgroundSubtractionManager
from signal_extraction_strategy import ErfParametrization

if __name__ == "__main__":
    # Loading data
    data = np.load("/kaggle/input/neurips-starter/output/data_train.npy")

    planet_id = 0
    time_steps = np.arange(0, 187)  # Time steps

    # Instantiate the manager with the initial strategy (ConstantBackgroundSubtraction)
    #background_manager = BackgroundSubtractionManager(ConstantBackgroundSubtraction())

    # Initialize the SignalExtractor with a constant background subtraction strategy
    extractor = SignalExtractor(QuadraticBackgroundSubtraction(), ErfParametrization())

    # Extract signal using the constant strategy
    data_normalized, quality_metric, a_vals, c_vals, t0_vals, sigma_vals, a_errs, *rest_errs = extractor.extract_signal(
        data, planet_id
    )
    print('Signal Spectrum (Const):', data_normalized)

    # Compute predictions and statistical errors
    preds = 2*np.abs(a_vals)[::-1]
    preds_err = 3*np.abs(a_errs)[::-1]