import scipy
import numpy as np

def extract_peak(targets, ipeak_low, ipeak_high, prominence=0.00005, width=25, rel_height=0.75):
    """
    Extracts and averages the peak from the target curves within the specified range using scipy's find_peaks.

    Parameters:
    - targets (ndarray): Array of true transit depth curves. Shape (n_curves, n_wavelengths).
    - ipeak_low (int): Start index of the peak region.
    - ipeak_high (int): End index of the peak region.
    - prominence (float): Required prominence of peaks to detect significant peaks.
    - width (int): Minimum width of peaks to consider (in index points).
    - rel_height (float): Relative height for measuring the width of the peak.

    Returns:
    - peak_avg (ndarray): The averaged peak profile across all target curves where a peak was detected.
    - extracted_peaks (list of ndarray): List of extracted peaks from each curve.
    """
    extracted_peaks = []
    peak_starts, peak_ends = [], []
    
    for target in targets:
        #print('Checking planet ...')
        # Extract the target data within the specified peak region
        peak_region = target[ipeak_low:ipeak_high]
        
        # Find peaks with the specified prominence and width parameters
        peaks, properties = scipy.signal.find_peaks(peak_region, prominence=prominence, width=width, rel_height=rel_height)
        #print('Found peaks:', len(peaks), peaks)
        
        # Check if any peaks were found in the region
        if len(peaks) > 0:
            # Take the most prominent peak if multiple peaks are detected
            main_peak_idx = peaks[np.argmax(properties['prominences'])]
            peak_start = int(properties['left_ips'][np.argmax(properties['prominences'])])
            peak_end = int(properties['right_ips'][np.argmax(properties['prominences'])])
            
            # Extract the main peak's region
            #print('Main peak range:', peak_start, peak_end)
            extracted_peak = peak_region[peak_start:peak_end]
            extracted_peaks.append(extracted_peak)
            peak_starts.append(peak_start)
            peak_ends.append(peak_end)

    # If any peaks were found, compute the average shape
    if extracted_peaks:
        # Rescale the peaks to the same area
        for ipeak,peak in enumerate(extracted_peaks):
            ground_level = 0.5*(peak[:2].mean() + peak[-2:].mean())
            peak -= ground_level
            peak = peak/peak.sum()
            extracted_peaks[ipeak] = peak
        
        # Pad the extracted peaks to the same length for averaging
        max_length = max(len(peak) for peak in extracted_peaks)
        padded_peaks = [np.pad(peak, (0, max_length - len(peak)), mode='constant', constant_values=np.nan) for peak in extracted_peaks]
        padded_peaks = np.array(padded_peaks)
        
        # Calculate the average peak, ignoring NaNs from padding
        peak_avg = np.nanmean(padded_peaks, axis=0)
        # Average time points
        peak_avg_start = int(np.array(peak_starts).mean())
        peak_avg_end = int(np.array(peak_ends).mean())
    else:
        peak_avg = np.array([])  # No peaks found

    return peak_avg, extracted_peaks, peak_avg_start, peak_avg_end

def check_peak_probability(preds, preds_err, ipeak_low, ipeak_high, alpha=0.05):
    """
    Checks for the presence of a peak in the specified region by comparing it
    to the surrounding regions and calculating a p-value based on z-scores.

    Parameters:
    - preds (ndarray): Array of predicted values across wavelengths.
    - preds_err (ndarray): Array of prediction errors (standard deviations) for each prediction point.
    - ipeak_low (int): Start index of the peak region.
    - ipeak_high (int): End index of the peak region.
    - alpha (float): Significance level for the p-value (default=0.05).

    Returns:
    - peak_present (bool): Whether a statistically significant peak is detected in the range.
    - peak_p_value (float): p-value indicating the probability of observing a peak due to noise.
    - contrast_score (float): Z-score quantifying the contrast between peak and surroundings.
    """

    # Define the width of the peak region
    peak_width = ipeak_high - ipeak_low
    if peak_width <= 0:
        raise ValueError("Invalid peak region; ipeak_high should be greater than ipeak_low.")
    
    # Define the surrounding regions
    left_low = max(0, ipeak_low - peak_width//2)  # Ensure indices are in range
    left_high = ipeak_low
    right_low = ipeak_high
    right_high = min(len(preds), ipeak_high + peak_width//2)

    # Extract values for peak and surrounding regions
    peak_region = preds[ipeak_low:ipeak_high]
    peak_region_err = preds_err[ipeak_low:ipeak_high]
    left_region = preds[left_low:left_high]
    left_region_err = preds_err[left_low:left_high]
    right_region = preds[right_low:right_high]
    right_region_err = preds_err[right_low:right_high]

    # Calculate mean and std for each region
    peak_mean = np.mean(peak_region)
    left_mean = np.mean(left_region)
    right_mean = np.mean(right_region)
    combined_surrounding_mean = (left_mean + right_mean) / 2

    peak_std = np.std(peak_region)
    surrounding_std = np.sqrt((np.std(left_region)**2 + np.std(right_region)**2) / 2)

    # Calculate contrast score (z-score) as the difference between peak and surrounding means
    contrast_score = (peak_mean - combined_surrounding_mean) / np.sqrt((peak_std**2 + surrounding_std**2) / 2)

    # Calculate the two-tailed p-value from the contrast score
    peak_p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(contrast_score)))

    # Determine if the peak is significant based on the alpha threshold
    peak_present = peak_p_value < alpha
    
    # Compute the peak amplitude
    peak_intensity = np.sum(peak_region - combined_surrounding_mean)

    return peak_present, peak_p_value, contrast_score, combined_surrounding_mean, peak_intensity

def replace_with_peak(preds, peak_shape, peak_intensity, surrounding_mean):
    """
    Replaces values in the 'preds' array with values from 'peak_shape' array
    only at the positions where 'peak_shape' is non-zero.
    
    Returns:
    - modified_preds (ndarray): A copy of 'preds' with values replaced in the peak region.
    """
    # Make a copy of preds to avoid modifying the original array
    modified_preds = np.copy(preds)
    
    # Compute the new peak values
    peak_values = peak_shape * peak_intensity/peak_shape.sum() + surrounding_mean
    
    # Replace values where peak_shape is non-zero
    modified_preds[peak_shape != 0] = peak_values[peak_shape != 0]
    
    return modified_preds