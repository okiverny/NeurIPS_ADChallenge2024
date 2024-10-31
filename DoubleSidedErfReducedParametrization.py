from signal_extraction_strategy import SignalExtractStrategy
import numpy as np
import warnings
import scipy
from helpers import erf_egress_func_reduced, erf_ingress_func_reduced, doublesided_erf_function_reduced

class DoubleSidedErfReducedParametrization(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, lambda_step=1, sigma_fix=None, results_incl=None):
        """
        Method for parametrization of the planet transition by double sided Erf function (simultanious fit).
        Allows binning the data in wavelength bins according to lambda_step.
        
        Parameters:
        - data: the data to be fitted (time x wavelength)
        - t_min, t_max: the time range for fitting
        - lambda_step: how many wavelength bins to combine together during fitting
        - sigma_fix: if there is a need to fix the sigma parameter in the fit (None: not fixed, float: fix value) - Does nothing here!
        - results_incl: results of wavelength-inclusive fit
        
        Returns:
        - Arrays of fit parameters: a, c=a, t0_ingress, t0_egress, sigma and their uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)

        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        ingress_init = (data.shape[0] // 3) - 2  # 60 --> tune this value
        egress_init = data.shape[0] - ingress_init
        midpoint = data.shape[0] // 2  # planet is at the middle

        # Initialize output values
        a = np.zeros(data.shape[1])
        a_err = np.zeros(data.shape[1])
        t0_ingress = np.zeros(data.shape[1])
        t0_ingress_err = np.zeros(data.shape[1])
        t0_egress = np.zeros(data.shape[1])
        t0_egress_err = np.zeros(data.shape[1])
        sigma = np.zeros(data.shape[1])
        sigma_err = np.zeros(data.shape[1])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:
            a_incl, c_incl, t0_ingress_incl, t0_egress_incl, sigma_incl, a_err_incl, c_err_incl, t0_ingress_err_incl, t0_egress_err_incl, sigma_err_incl = results_incl
            a.fill(a_incl[0])
            t0_ingress.fill(t0_ingress_incl[0])
            t0_egress.fill(t0_egress_incl[0])
            sigma.fill(sigma_incl[0])
            a_err.fill(a_err_incl[0])
            t0_ingress_err.fill(t0_ingress_err_incl[0])
            t0_egress_err.fill(t0_egress_err_incl[0])
            sigma_err.fill(sigma_err_incl[0])

        # Loop over wavelength and fit the Erf function to normalized data, using lambda_step
        for ilambda in range(0, data.shape[1] - lambda_step + 1, lambda_step):
            # Average the data over the wavelength step
            y = np.mean(data[t_min:t_max, ilambda:ilambda + lambda_step], axis=1)

            try:
                # Suppress OptimizeWarning and fit the erf function to the averaged data
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.optimize.OptimizeWarning)  # Treat OptimizeWarning as an exception

                    # Fit the erf function to the averaged data
                    pars_init = np.array([0, ingress_init, egress_init, 2]) # Initial guess for parameters
                    popt, pcov = scipy.optimize.curve_fit(doublesided_erf_function_reduced, t, y, p0=pars_init, maxfev=800)
                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))
                    
            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[3]>10:
                print(f"Fit is bad for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[1]:
                    # Store the best fit values
                    a[ilambda + i] = popt[0]
                    t0_ingress[ilambda + i] = popt[1]
                    t0_egress[ilambda + i] = popt[2]
                    sigma[ilambda + i] = popt[3]

                    # Store the parameter errors
                    a_err[ilambda + i] = perr[0]
                    t0_ingress_err[ilambda + i] = perr[1]
                    t0_egress_err[ilambda + i] = perr[2]
                    sigma_err[ilambda + i] = perr[3]

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[1] % lambda_step
        if remaining_bins > 0:
            # Average the data for the remaining bins
            y = np.mean(data[t_min:t_max, -remaining_bins:], axis=1)

            try:
                # Fit the erf function to the remaining data
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.optimize.OptimizeWarning)

                    # Fit the erf function to the averaged data
                    pars_init = np.array([0, 1, ingress_init, egress_init, 2]) # Initial guess for parameters
                    popt, pcov = scipy.optimize.curve_fit(doublesided_erf_function_reduced, t, y, p0=pars_init, maxfev=800)
                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))

            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[3]>10:
                print(f"Fit is bad for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                a[-remaining_bins + i] = popt[0]
                t0_ingress[-remaining_bins + i] = popt[1]
                t0_egress[-remaining_bins + i] = popt[2]
                sigma[-remaining_bins + i] = popt[3]

                a_err[-remaining_bins + i] = perr[0]
                t0_ingress_err[-remaining_bins + i] = perr[1]
                t0_egress_err[-remaining_bins + i] = perr[2]
                sigma_err[-remaining_bins + i] = perr[3]

        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(a), np.array(a), np.array(t0_ingress), np.array(t0_egress), np.array(sigma), np.array(a_err), np.array(a_err), np.array(t0_ingress_err), np.array(t0_egress_err), np.array(sigma_err)
    