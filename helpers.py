from scipy.signal import savgol_filter
import scipy
import numpy as np

def erf_func(t, *args):
    a, c, t0, sigma = args
    return a * scipy.special.erf((t - t0)/sigma) + c

def erf_egress_func(t, *args):
    a, c, t0_ingress, t0_egress, sigma = args
    return a * scipy.special.erf((t - t0_egress)/sigma) + c

def erf_ingress_func(t, *args):
    a, c, t0_ingress, t0_egress, sigma = args
    return -a * scipy.special.erf((t - t0_ingress)/sigma) + c

def doublesided_erf_function(comboData, *args):
    a, c, t0_ingress, t0_egress, sigma = args

    # single data reference passed in, extract separate data
    midpoint = len(comboData)//2
    data_left = comboData[:midpoint] # first data
    data_right = comboData[midpoint:] # second data

    result1 = erf_ingress_func(data_left, a, c, t0_ingress, t0_egress, sigma)
    result2 = erf_egress_func(data_right, a, c, t0_ingress, t0_egress, sigma)

    return np.append(result1, result2)

def linear_func(t, *args):
    a, c = args
    return a * t + c

def quadratic_func(t, *args):
    a, b, c = args
    return a*t*t + b*t + c

def calc_chisquare(meas, sigma, fit):
    diff = pow(meas-fit, 2.)
    test_statistic = (diff / pow(sigma,2.)).sum()
    return test_statistic

def smooth_data(data, window_size):
    return savgol_filter(data, window_size, 3)  # window size 51, polynomial order 3

def optimize_breakpoint(data, initial_breakpoint, window_size=20, buffer_size=8, smooth_window=11):
    best_breakpoint = initial_breakpoint
    best_score = float("-inf")
    smoothed_data = smooth_data(data, smooth_window)

    for i in range(-window_size, window_size):
        new_breakpoint = initial_breakpoint + i
        region1 = data[: new_breakpoint - buffer_size]
        region2 = data[
            new_breakpoint
            + buffer_size : len(data)
            - new_breakpoint
            - buffer_size
        ]
        region3 = data[len(data) - new_breakpoint + buffer_size :]

        # calc on smoothed data
        breakpoint_region1 = smoothed_data[new_breakpoint - buffer_size: new_breakpoint + buffer_size]
        breakpoint_region2 = smoothed_data[-(new_breakpoint + buffer_size): -(new_breakpoint - buffer_size)]

        mean_diff = abs(np.mean(region1) - np.mean(region2)) + abs(
            np.mean(region2) - np.mean(region3)
        )
        var_sum = np.var(region1) + np.var(region2) + np.var(region3)
        range_at_breakpoint1 = (np.max(breakpoint_region1) - np.min(breakpoint_region1))
        range_at_breakpoint2 = (np.max(breakpoint_region2) - np.min(breakpoint_region2))

        mean_range_at_breakpoint = (range_at_breakpoint1 + range_at_breakpoint2) / 2

        score = mean_diff - 0.5 * var_sum + mean_range_at_breakpoint

        if score > best_score:
            best_score = score
            best_breakpoint = new_breakpoint

    return best_breakpoint

# Bootstrap function for better errors assessment
def bootstrap_signal_extraction(extractor, data, planet_id, n_bootstrap=100):
    bootstrap_preds = []
    bootstrap_errors = []

    for _ in range(n_bootstrap):
        # Resample data with replacement (bootstrap sample)
        bootstrap_sample = np.random.choice(data.shape[1], size=data.shape[1], replace=True)
        data_bootstrap = data[:, bootstrap_sample]

        # Extract signal from the bootstrap sample
        data_normalized, quality_metric, a_vals, c_vals, t0_vals, sigma_vals, a_errs, *rest_errs = extractor.extract_signal(
            data_bootstrap, planet_id
        )

        # Compute predictions and errors
        preds = 2 * np.abs(a_vals)[::-1]
        preds_err = 3 * np.abs(a_errs)[::-1]

        # Approximate w1 value
        preds = np.concatenate(([preds[3:8].mean()], preds))
        preds_err = np.concatenate(([preds_err[3:8].mean()], preds_err))

        # Store results
        bootstrap_preds.append(preds)
        bootstrap_errors.append(preds_err)

    # Convert lists to arrays for easier manipulation
    bootstrap_preds = np.array(bootstrap_preds)
    bootstrap_errors = np.array(bootstrap_errors)

    # Calculate mean predictions and error estimates (standard deviation or confidence intervals)
    preds_mean = np.mean(bootstrap_preds, axis=0)
    preds_err_mean = np.std(bootstrap_errors, axis=0)  # Standard deviation as an estimate of error

    return preds_mean, preds_err_mean