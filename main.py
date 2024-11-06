import numpy as np
import pandas as pd
from signal_extraction import SignalExtractor
from background_subtraction import ConstantBackgroundSubtraction, LinearBackgroundSubtraction, QuadraticBackgroundSubtraction, CubicBackgroundSubtraction, BackgroundSubtractionManager
from signal_extraction_strategy import ErfParametrization, DoubleSidedErfParametrization, ZFitParametrization
from peak_finding import extract_peak, check_peak_probability, replace_with_peak

if __name__ == "__main__":
    # Loading data
    data = np.load("/kaggle/input/neurips-starter/output/data_train.npy")
    data_FGS = np.load("/kaggle/input/neurips-starter/output/data_train_FGS.npy")

    # Optimize data_FGS representation. It has (planet_id, time, x, y) shape. The slice 9:23 provides the best senisitivity on the train data.
    data_FGS = data_FGS[:, :, 9:23, :].sum(axis=2)[:,:, np.newaxis,:]

    ############ Finding Peak 28-65 (small improvement) #################
    # Leading the targets from train data. These are used for peaks finding in the predictions.
    train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2024/train_labels.csv', index_col='planet_id')

    peak_low, peak_high = 10, 75
    peak_avg, extracted_peaks, peak_start, peak_end = extract_peak(train_labels.values, peak_low, peak_high, width=35, prominence=0.0003, rel_height=0.75)
    print('Number of peaks settled within 130 and 270:', len(extracted_peaks))
    peak_28_65_shape = np.concatenate([np.zeros(peak_low+peak_start), peak_avg, np.zeros(283-peak_start-peak_low-len(peak_avg))]).clip(0)
    peak_28_65_low, peak_28_65_high = peak_low+peak_start, peak_low+peak_end # = 28, 65


    planet_id = 0
    time_steps = np.arange(0, 187)  # Time steps

    # Instantiate the manager with the initial strategy (ConstantBackgroundSubtraction)
    #background_manager = BackgroundSubtractionManager(ConstantBackgroundSubtraction())

    # Initialize the SignalExtractor with a constant background subtraction strategy
    extractor = SignalExtractor(QuadraticBackgroundSubtraction(), ErfParametrization())

    # Extract signal using the constant strategy
    data_normalized, quality_metric, a_vals, c_vals, t0_vals, sigma_vals, a_errs, *rest_errs = extractor.extract_signal(
        data, data_FGS, planet_id
    )
    print('Signal Spectrum (Const):', data_normalized)

    # Compute predictions and statistical errors
    preds = 2*np.abs(a_vals)
    preds_err = 3*np.abs(a_errs)