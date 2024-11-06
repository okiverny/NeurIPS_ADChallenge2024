import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from calibrating import Calibrator
from signal_extraction import SignalExtractor
from background_subtraction import ConstantBackgroundSubtraction, LinearBackgroundSubtraction, QuadraticBackgroundSubtraction, CubicBackgroundSubtraction, BackgroundSubtractionManager
from signal_extraction_strategy import ErfParametrization, DoubleSidedErfParametrization, ZFitParametrization
from peak_finding import extract_peak, check_peak_probability, replace_with_peak
from helpers import evaluate_flatness

if __name__ == "__main__":
    # Loading data
    data = np.load("/kaggle/input/neurips-starter/output/data_test.npy")
    data_FGS = np.load("/kaggle/input/neurips-starter/output/data_test_FGS.npy")

    ############ Data Calibration #################
    OUTPUT_DIR="output"

    calibrator = Calibrator(
        is_test=True,    # False: training dataset; True: test dataset
        data_dir="/kaggle/input/ariel-data-challenge-2024",
        output_dir="output",
        c_lib_path="/kaggle/input/adc-calibration-library/c_lib.so",
        num_workers=8,
        time_binning_freq=30,
        #first_n_files=8,    # Use this option if you only want to calibrate a few files for testing
        verbose=2,    # 0: None, 1: progress bar, 2: print
    )
    calibrator.calibrate()
    calibrator.concatenate_files()
    del calibrator

    # Optimize data_FGS representation. It has (planet_id, time, x, y) shape. The slice 9:23 provides the best senisitivity on the train data.
    data_FGS = data_FGS[:, :, 9:23, :].sum(axis=2)[:,:, np.newaxis,:]

    ############ Finding Peak 85-124 (small improvement) #################
    # Leading the targets from train data. These are used for peaks finding in the predictions.
    train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2024/train_labels.csv', index_col='planet_id')

    peak_low, peak_high = 76, 140
    peak_avg, extracted_peaks, peak_start, peak_end = extract_peak(train_labels.values, peak_low, peak_high, prominence=0.0003)
    print('Number of peaks settled within 76 and 140:', len(extracted_peaks))
    peak_85_124_shape = np.concatenate([np.zeros(peak_low+peak_start), peak_avg, np.zeros(283-peak_start-peak_low-len(peak_avg))]).clip(0)
    peak_85_124_low, peak_85_124_high = peak_low+peak_start, peak_low+peak_end # = 85, 124

    ############ Main loop over planets #################
    # Initialize the SignalExtractor with a constant background subtraction strategy
    extractor = SignalExtractor(QuadraticBackgroundSubtraction(), ErfParametrization())
    
    predictions, sigmas = [], []
    for planet_id in range(data.shape[0]):
        print('====== Processing planet:', planet_id)

        ########## Results of 15-lambdas fits
        _, _, preds, _, _, _, preds_err, *_ = extractor.extract_signal(
            data, data_FGS, planet_id, lambda_step=15
        )
        # Sometimes the extrapolation to FGS performs better
        if np.abs(preds[0] - preds[3:9].mean()) > 2*0.0001:
            print('FGS does not look good! Replacing it ...')
            preds[0] = preds[3:9].mean()
            preds_err[0] = preds_err[3:9].mean()

        ########## Results of inclusive lambda fits
        _, _, preds_incl, _, _, _, preds_err_incl, *_ = extractor.extract_signal(
            data, data_FGS, planet_id, lambda_step=data.shape[2]
        )
        preds_incl[0], preds_err_incl[0] = preds_incl[3:9].mean(), preds_err_incl[3:9].mean()


        ######### Error adjustment (Confidence Intervals)
        preds_err = (preds_err*4.2)
        preds_err_incl = (preds_err_incl*4.2)

        # Last bins alignment (sometimes last bin(s) can be significantly off due to lower statistics)
        if preds[-7]==np.max(preds) or preds[-7]==np.min(preds): # last bin
            preds[-15:] = preds_incl[-7]
        if preds[-22]==np.min(preds) or preds[-22]==np.max(preds):
            preds[-30:-15] = preds_incl[-22]

        # Check specific peaks: 85-124
        peak_85_124_present, peak_85_124_p_value, contrast_85_124_score, surrounding_mean_85_124, peak_85_124_intensity = check_peak_probability(preds, preds_err, peak_85_124_low, peak_85_124_high)
        print("Is there a statistically significant peak?", peak_85_124_present, "  Peak p-value:", peak_85_124_p_value)

        ######### Check flatness
        MinMax = preds.max() - preds.min()
        chi2_stat, p_value = evaluate_flatness(preds, preds_err, preds_incl, preds_err_incl)
        if p_value < 0.003:
            print("The spectrum shows significant dependence on wavelength (reject flatness).")
            preds_err = preds_err.clip(0.00008)
        
            # Adjustment of known peaks
            if peak_85_124_present and peak_85_124_intensity>0:
                preds = replace_with_peak(preds, peak_85_124_shape, peak_85_124_intensity, surrounding_mean_85_124)
            
            ### smoothing
            if p_value < 0.00005:
                preds = savgol_filter(preds, 35, 3)
        else:
            print(f"The spectrum is consistent with a flat prediction (accept flatness). P-value: {p_value:.5f}")
            preds = preds_incl
            preds_err = 1.6*preds_err_incl.clip(0.000025)
        
            # Adjustment of known peaks in the inclusive spectre
            if peak_85_124_p_value<0.083 and peak_85_124_intensity>0:
                preds = replace_with_peak(preds, peak_85_124_shape, peak_85_124_intensity, surrounding_mean_85_124)
        
        # Store predictions
        predictions.append( preds )
        sigmas.append( preds_err )
    
    # Convert the predictions to an array
    predictions = np.array(predictions)
    sigmas = np.array(sigmas)