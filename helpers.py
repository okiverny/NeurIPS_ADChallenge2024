from scipy.signal import savgol_filter
from scipy.stats import chisquare
from scipy.stats import chi2
import scipy
import numpy as np

# Function to calculate uncertainty
def find_uncertainty(func, best_value, chi2_min, step=0.000001, ndf=1):
    chi2_target = chi2_min + 1/ndf  # Target chi2 value

    # Initialize values slightly above and below the minimum MW value
    mu_high = best_value + step
    mu_low = best_value - step

    # Increase the range until the target chi2 value is reached
    while func(mu_high) <= chi2_target:
        mu_high += step
    while func(mu_low) <= chi2_target:
        mu_low -= step

    # Calculate uncertainty as half the difference between the MW values
    uncertainty = (mu_high - mu_low) / 2
    return uncertainty

def erf_func(t, *args):
    a, c, t0, sigma = args
    return a * scipy.special.erf((t - t0)/sigma) + c

def erf_egress_func(t, a, c, t0_ingress, t0_egress, sigma):
    return a * scipy.special.erf((t - t0_egress)/sigma) + c

def erf_ingress_func(t, a, c, t0_ingress, t0_egress, sigma):
    return -a * scipy.special.erf((t - t0_ingress)/sigma) + c

def doublesided_erf_function(comboData, a, c, t0_ingress, t0_egress, sigma):
    # single data reference passed in, extract separate data
    midpoint = len(comboData)//2
    data_left = comboData[:midpoint] # first data
    data_right = comboData[midpoint:] # second data
    result1 = erf_ingress_func(data_left, a, c, t0_ingress, t0_egress, sigma)
    result2 = erf_egress_func(data_right, a, c, t0_ingress, t0_egress, sigma)
    return np.append(result1, result2)

# Define your custom fit function: a parabola with parameters a, b, and c
# b is the fitted value and c the error
def chi2_interpol_func(x, a, b, c):
    return a + (x - b) * (x - b) / c**2

# Background subtraction functions
def linear_func(t, *args):
    a, c = args
    return a * t + c

def quadratic_func(t, a, b, c):
    #a, b, c = args
    return a*t*t + b*t + c

def pol3_func(t, *args):
    d, a, b, c = args
    return d*t*t*t + a*t*t + b*t + c

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


def evaluate_flatness(preds, preds_err, preds_incl, preds_err_incl):
    """
    Evaluate whether the predictions show a significant dependence on wavelength 
    or if they are consistent with a flat spectrum.
    
    Parameters:
    preds: numpy array of predicted values (size 282)
    preds_err: numpy array of predicted errors (size 282)
    preds_incl: numpy array of flat predicted values (size 282, wavelength-inclusive)
    preds_err_incl: numpy array of flat predicted errors (size 282, wavelength-inclusive)
    
    Returns:
    chi2: Chi-square statistic for testing flatness
    p_value: p-value indicating whether to reject the null hypothesis of flatness
    """

    # Chi-square test: (obs - exp)^2 / err^2
    # We combine errors from the predictions and the flat prediction in quadrature
    combined_err = np.sqrt(preds_err**2 + preds_err_incl**2)

    # Compute chi-squared statistic comparing preds (observed) to preds_incl (expected flat model)
    chi2_stat = np.sum((preds - preds_incl)**2 / combined_err**2)

    # Degrees of freedom: number of data points minus 1 (since flat model has one parameter)
    dof = len(preds) - 1

    # Compute the p-value from the chi-square statistic and degrees of freedom
    #p_value = chisquare(preds, preds_incl, ddof=dof)[1]
    p_value = chi2.sf(chi2_stat, dof)  # Use survival function to get the p-value

    return chi2_stat/dof, p_value