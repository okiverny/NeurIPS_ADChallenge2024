from abc import ABC, abstractmethod
import numpy as np
import scipy
import warnings
import tqdm
from iminuit import Minuit, cost
from helpers import erf_func, erf_egress_func, erf_ingress_func, doublesided_erf_function
from zfit_helpers import fit_planet, get_pdf, PlanetTimePDF
from minuit_helpers import fit_transit, polynomial

# Strategy interface
class SignalExtractStrategy(ABC):
    """This is the interface that declares operations common to all signal extraction algorithms/methods."""
    @abstractmethod
    def parametrize_data(self, data, t_min, t_max, lambda_step=1, sigma_fix=None, results_incl=None):
        pass

# Concrete Strategies
class ErfParametrization(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, lambda_step=1, sigma_fix=None, results_incl=None):
        """
        Method for parametrization of the planet transition by Erf function.
        Allows binning the data in wavelength bins according to lambda_step.
        
        Parameters:
        - data: the data to be fitted (time x wavelength)
        - t_min, t_max: the time range for fitting
        - lambda_step: how many wavelength bins to combine together during fitting
        - sigma_fix: if there is a need to fix the sigma parameter in the fit (None: not fixed, float: fix value)
        - results_incl: results of wavelength-inclusive fit
        
        Returns:
        - Arrays of fit parameters: a, c, t0, sigma and their uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)
        
        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        ingress_init = (data.shape[0] // 3) - 2  # 60 --> tune this value
        transit_init = ingress_init if (ingress_init>t_min and ingress_init<t_max) else data.shape[0]-ingress_init

        # Initialize output values
        a = np.zeros(data.shape[1])
        a_err = np.zeros(data.shape[1])
        c = np.zeros(data.shape[1])
        c_err = np.zeros(data.shape[1])
        t0 = np.zeros(data.shape[1])
        t0_err = np.zeros(data.shape[1])
        sigma = np.zeros(data.shape[1])
        sigma_err = np.zeros(data.shape[1])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:
            a_incl, c_incl, t0_incl, sigma_incl, a_err_incl, c_err_incl, t0_err_incl, sigma_err_incl = results_incl
            a.fill(a_incl[0])
            c.fill(c_incl[0])
            t0.fill(t0_incl[0])
            sigma.fill(sigma_incl[0])
            a_err.fill(a_err_incl[0])
            c_err.fill(c_err_incl[0])
            t0_err.fill(t0_err_incl[0])
            sigma_err.fill(sigma_err_incl[0])
        
        # Loop over wavelength and fit the Erf function to normalized data, using lambda_step
        for ilambda in tqdm.tqdm(range(0, data.shape[1] - lambda_step + 1, lambda_step)):
            # Average the data over the wavelength step
            y = np.mean(data[t_min:t_max, ilambda:ilambda + lambda_step], axis=1)

            try:
                # Suppress OptimizeWarning and fit the erf function to the averaged data
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.optimize.OptimizeWarning)  # Treat OptimizeWarning as an exception

                    # Fit the erf function to the averaged data
                    pars_init = np.array([0, 1, transit_init] if sigma_fix else [0, 1, transit_init, 2]) # Initial guess for parameters
                    if not sigma_fix:
                        popt, pcov = scipy.optimize.curve_fit(erf_func, t, y, p0=pars_init, maxfev=800)
                    else:   # use lambda function to fix sigma parameter to sigma_fix
                        popt, pcov = scipy.optimize.curve_fit(lambda x, a, c, t0: erf_func(x, a, c, t0, sigma_fix), t, y, p0=pars_init, maxfev=800)

                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))

            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_incl[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[3]>10:
                print(f"Fit is bad for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_incl[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[1]:
                    # Store the best fit values
                    a[ilambda + i] = popt[0]
                    c[ilambda + i] = popt[1]
                    t0[ilambda + i] = popt[2]
                    sigma[ilambda + i] = popt[3] if not sigma_fix else sigma_fix

                    # Store the parameter errors
                    a_err[ilambda + i] = perr[0]
                    c_err[ilambda + i] = perr[1]
                    t0_err[ilambda + i] = perr[2]
                    sigma_err[ilambda + i] = perr[3] if not sigma_fix else 0.0001

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[1] % lambda_step
        if remaining_bins > 0:
            # Average the data for the remaining bins
            y = np.mean(data[t_min:t_max, -remaining_bins:], axis=1)

            try:
                # Fit the erf function to the remaining data
                with warnings.catch_warnings():
                    warnings.simplefilter("error", scipy.optimize.OptimizeWarning)

                    # Fit the erf function to the remaining data
                    pars_init = np.array([0, 1, transit_init] if sigma_fix else [0, 1, transit_init, 2])
                    if not sigma_fix:
                        popt, pcov = scipy.optimize.curve_fit(erf_func, t, y, p0=pars_init, maxfev=800)
                    else:   # use lambda function to fix sigma parameter to sigma_fix
                        popt, pcov = scipy.optimize.curve_fit(lambda x, a, c, t0: erf_func(x, a, c, t0, sigma_fix), t, y, p0=pars_init, maxfev=800)

                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))

            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_incl[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[3]>10:
                print(f"Fit is bad for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_incl[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                a[-remaining_bins + i] = popt[0]
                c[-remaining_bins + i] = popt[1]
                t0[-remaining_bins + i] = popt[2]
                sigma[-remaining_bins + i] = popt[3] if not sigma_fix else sigma_fix

                a_err[-remaining_bins + i] = perr[0]
                c_err[-remaining_bins + i] = perr[1]
                t0_err[-remaining_bins + i] = perr[2]
                sigma_err[-remaining_bins + i] = perr[3] if not sigma_fix else 0.0001

        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(a), np.array(c), np.array(t0), np.array(sigma), np.array(a_err), np.array(c_err), np.array(t0_err), np.array(sigma_err)
    
class DoubleSidedErfParametrization(SignalExtractStrategy):
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
        - Arrays of fit parameters: a, c, t0_ingress, t0_egress, sigma and their uncertainties
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
        c = np.zeros(data.shape[1])
        c_err = np.zeros(data.shape[1])
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
            c.fill(c_incl[0])
            t0_ingress.fill(t0_ingress_incl[0])
            t0_egress.fill(t0_egress_incl[0])
            sigma.fill(sigma_incl[0])
            a_err.fill(a_err_incl[0])
            c_err.fill(c_err_incl[0])
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
                    pars_init = np.array([0, 1, ingress_init, egress_init, 2]) # Initial guess for parameters
                    popt, pcov = scipy.optimize.curve_fit(doublesided_erf_function, t, y, p0=pars_init, maxfev=800)
                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))
                    
            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[4]>10:
                print(f"Fit is bad for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[1]:
                    # Store the best fit values
                    a[ilambda + i] = popt[0]
                    c[ilambda + i] = popt[1]
                    t0_ingress[ilambda + i] = popt[2]
                    t0_egress[ilambda + i] = popt[3]
                    sigma[ilambda + i] = popt[4]

                    # Store the parameter errors
                    a_err[ilambda + i] = perr[0]
                    c_err[ilambda + i] = perr[1]
                    t0_ingress_err[ilambda + i] = perr[2]
                    t0_egress_err[ilambda + i] = perr[3]
                    sigma_err[ilambda + i] = perr[4]

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
                    popt, pcov = scipy.optimize.curve_fit(doublesided_erf_function, t, y, p0=pars_init, maxfev=800)
                    # Uncertainties
                    perr = np.sqrt(np.diag(pcov))

            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[4]>10:
                print(f"Fit is bad for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0005, 0.0005, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                a[-remaining_bins + i] = popt[0]
                c[-remaining_bins + i] = popt[1]
                t0_ingress[-remaining_bins + i] = popt[2]
                t0_egress[-remaining_bins + i] = popt[3]
                sigma[-remaining_bins + i] = popt[4]

                a_err[-remaining_bins + i] = perr[0]
                c_err[-remaining_bins + i] = perr[1]
                t0_ingress_err[-remaining_bins + i] = perr[2]
                t0_egress_err[-remaining_bins + i] = perr[3]
                sigma_err[-remaining_bins + i] = perr[4]

        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(a), np.array(c), np.array(t0_ingress), np.array(t0_egress), np.array(sigma), np.array(a_err), np.array(c_err), np.array(t0_ingress_err), np.array(t0_egress_err), np.array(sigma_err)
    
class DSErfMinuitParametrization(SignalExtractStrategy):
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
        - Arrays of fit parameters: a, c, t0_ingress, t0_egress, sigma and their uncertainties
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
        c = np.zeros(data.shape[1])
        c_err = np.zeros(data.shape[1])
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
            c.fill(c_incl[0])
            t0_ingress.fill(t0_ingress_incl[0])
            t0_egress.fill(t0_egress_incl[0])
            sigma.fill(sigma_incl[0])
            a_err.fill(a_err_incl[0])
            c_err.fill(c_err_incl[0])
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
                    initial_params = {'a': 0, 'c':1, 't0_ingress':ingress_init, 't0_egress':egress_init, 'sigma':2}
                    ye = np.concatenate( (y[0:ingress_init-8], y[egress_init+8:len(y)]) ).std()
                    ye = np.ones_like(y)*ye
                    cost_function = cost.LeastSquares(t, y, ye, doublesided_erf_function)
                    m = Minuit(cost_function,  **initial_params)
                    m.migrad()
                    m.hesse()
                    popt = (m.values['a'], m.values['c'], m.values['t0_ingress'], m.values['t0_egress'], m.values['sigma'])
                    perr = (m.errors['a'], m.errors['c'], m.errors['t0_ingress'], m.errors['t0_egress'], m.errors['sigma'])
                    
            except (RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0001, 0.0001, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[4]>10:
                print(f"Fit is bad for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0001, 0.0001, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[1]:
                    # Store the best fit values
                    a[ilambda + i] = popt[0]
                    c[ilambda + i] = popt[1]
                    t0_ingress[ilambda + i] = popt[2]
                    t0_egress[ilambda + i] = popt[3]
                    sigma[ilambda + i] = popt[4]

                    # Store the parameter errors
                    a_err[ilambda + i] = perr[0]
                    c_err[ilambda + i] = perr[1]
                    t0_ingress_err[ilambda + i] = perr[2]
                    t0_egress_err[ilambda + i] = perr[3]
                    sigma_err[ilambda + i] = perr[4]

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[1] % lambda_step
        if remaining_bins > 0:
            # Average the data for the remaining bins
            y = np.mean(data[t_min:t_max, -remaining_bins:], axis=1)

            try:
                # Fit the erf function to the remaining data
                with warnings.catch_warnings():
                    initial_params = {'a': 0, 'c':1, 't0_ingress':ingress_init, 't0_egress':egress_init, 'sigma':2}
                    ye = np.concatenate( (y[0:ingress_init-8], y[egress_init+8:len(y)]) ).std()
                    ye = np.ones_like(y)*ye
                    cost_function = cost.LeastSquares(t, y, ye, doublesided_erf_function)
                    m = Minuit(cost_function,  **initial_params)
                    m.migrad()
                    m.hesse()
                    popt = (m.values['a'], m.values['c'], m.values['t0_ingress'], m.values['t0_egress'], m.values['sigma'])
                    perr = (m.errors['a'], m.errors['c'], m.errors['t0_ingress'], m.errors['t0_egress'], m.errors['sigma'])

            except (RuntimeError):
                print(f"Fit failed for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0001, 0.0001, 0.0005, 0.0005, 0.0005)

            # Protection against 'bad' fits
            if popt[0]>0.5 or popt[4]>10:
                print(f"Fit is bad for the remaining wavelength bins. Returning inclusive values for this range.")
                popt = (a_incl[0], c_incl[0], t0_ingress[0], t0_egress[0], sigma_incl[0]) if results_incl else (0.003, 0.003, 60.0, 126.0, 1.7)
                perr = (a_err_incl[0], c_err_incl[0], t0_ingress_err_incl[0], t0_ingress_err_incl[0], sigma_err_incl[0]) if results_incl else (0.0001, 0.0001, 0.0005, 0.0005, 0.0005)

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                a[-remaining_bins + i] = popt[0]
                c[-remaining_bins + i] = popt[1]
                t0_ingress[-remaining_bins + i] = popt[2]
                t0_egress[-remaining_bins + i] = popt[3]
                sigma[-remaining_bins + i] = popt[4]

                a_err[-remaining_bins + i] = perr[0]
                c_err[-remaining_bins + i] = perr[1]
                t0_ingress_err[-remaining_bins + i] = perr[2]
                t0_egress_err[-remaining_bins + i] = perr[3]
                sigma_err[-remaining_bins + i] = perr[4]

        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(a), np.array(c), np.array(t0_ingress), np.array(t0_egress), np.array(sigma), np.array(a_err), np.array(c_err), np.array(t0_ingress_err), np.array(t0_egress_err), np.array(sigma_err)
    

class ZFitParametrization(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, planet_id, lambda_step=1, sigma_fix=None, results_incl=None):
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
        - Arrays of fit parameters: frac, center, width, tan, c0 and their uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)
        floating_pars = False if results_incl else True

        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        ingress_init = (data.shape[1] // 3) - 2  # 60 --> tune this value
        egress_init = data.shape[1] - ingress_init

        # Initialize output values
        frac = np.zeros(data.shape[2])
        frac_err = np.zeros(data.shape[2])
        center = np.zeros(data.shape[2])
        center_err = np.zeros(data.shape[2])
        width = np.zeros(data.shape[2])
        width_err = np.zeros(data.shape[2])
        tan = np.zeros(data.shape[2])
        tan_err = np.zeros(data.shape[2])
        c0 = np.zeros(data.shape[2])
        c0_err = np.zeros(data.shape[2])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:            
            frac.fill(results_incl.params['frac_0']['value'])
            center.fill(results_incl.params['center_0']['value'])
            width.fill(results_incl.params['width_0']['value'])
            tan.fill(results_incl.params['tan_0']['value'])
            c0.fill(results_incl.params['c0_0']['value'])
            frac_err.fill(results_incl.params['frac_0']['hesse']['error'])
            center_err.fill(results_incl.params['center_0']['hesse']['error'])
            width_err.fill(results_incl.params['width_0']['hesse']['error'])
            tan_err.fill(results_incl.params['tan_0']['hesse']['error'])
            c0_err.fill(results_incl.params['c0_0']['hesse']['error'])

        # Loop over wavelength and fit the zfit function
        for ilambda in range(0, data.shape[2] - lambda_step + 1, lambda_step):

            lambdas = (ilambda, ilambda + lambda_step)
            params = results_incl.params if results_incl else None
            obs, sec_pdf, sec_result = fit_planet(data, planet_id, lambdas=lambdas, params=params, floating=floating_pars)

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[2]:
                    # Store the best fit values
                    frac[ilambda + i] = sec_result.params['frac_0']['value']
                    c0[ilambda + i] = sec_result.params['c0_0']['value']
                    if floating_pars:
                        center[ilambda + i] = sec_result.params['center_0']['value']
                        width[ilambda + i] = sec_result.params['width_0']['value']
                        tan[ilambda + i] = sec_result.params['tan_0']['value']

                    # Store the parameter errors
                    frac_err[ilambda + i] = sec_result.params['frac_0']['hesse']['error']
                    c0_err[ilambda + i] = sec_result.params['c0_0']['hesse']['error']
                    if floating_pars:
                        center_err[ilambda + i] = sec_result.params['center_0']['hesse']['error']
                        width_err[ilambda + i] = sec_result.params['width_0']['hesse']['error']
                        tan_err[ilambda + i] = sec_result.params['tan_0']['hesse']['error']
                    

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[2] % lambda_step
        if remaining_bins > 0:
            # Average the data for the remaining bins
            lambdas = (data.shape[2]-remaining_bins, data.shape[2])
            params = results_incl.params if results_incl else None
            obs, sec_pdf, sec_result = fit_planet(data, planet_id, lambdas=lambdas, params=params, floating=floating_pars)

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                frac[-remaining_bins + i] = sec_result.params['frac_0']['value']
                c0[-remaining_bins + i] = sec_result.params['c0_0']['value']
                if floating_pars:
                    center[-remaining_bins + i] = sec_result.params['center_0']['value']
                    width[-remaining_bins + i] = sec_result.params['width_0']['value']
                    tan[-remaining_bins + i] = sec_result.params['tan_0']['value']

                frac_err[-remaining_bins + i] = sec_result.params['frac_0']['hesse']['error']
                c0_err[-remaining_bins + i] = sec_result.params['c0_0']['hesse']['error']
                if floating_pars:
                    center_err[-remaining_bins + i] = sec_result.params['center_0']['hesse']['error']
                    width_err[-remaining_bins + i] = sec_result.params['width_0']['hesse']['error']
                    tan_err[-remaining_bins + i] = sec_result.params['tan_0']['hesse']['error']
                
        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(frac), np.array(center), np.array(width), np.array(tan), np.array(c0), np.array(frac_err), np.array(center_err), np.array(width_err), np.array(tan_err), np.array(c0_err), sec_result
    
class MinuitParametrization(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, planet_id, lambda_step=1, sigma_fix=None, results_incl=None):
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
        - Arrays of fit parameters: frac, center, width, tan, c0 and their uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)
        fix_params = ['tan', 'center', 'width'] if results_incl else None
        new_params = results_incl['values'].to_dict() if results_incl else None

        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        ingress_init = (data.shape[1] // 3) - 2  # 60 --> tune this value
        egress_init = data.shape[1] - ingress_init

        # Initialize output values
        frac = np.zeros(data.shape[2])
        frac_err = np.zeros(data.shape[2])
        center = np.zeros(data.shape[2])
        center_err = np.zeros(data.shape[2])
        width = np.zeros(data.shape[2])
        width_err = np.zeros(data.shape[2])
        tan = np.zeros(data.shape[2])
        tan_err = np.zeros(data.shape[2])
        c0 = np.zeros(data.shape[2])
        c0_err = np.zeros(data.shape[2])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:
            frac.fill(results_incl['values']['frac'])
            center.fill(results_incl['values']['center'])
            width.fill(results_incl['values']['width'])
            tan.fill(results_incl['values']['tan'])
            c0.fill(results_incl['values']['c0'])
            frac_err.fill(results_incl['errors']['frac'])
            center_err.fill(results_incl['errors']['center'])
            width_err.fill(results_incl['errors']['width'])
            tan_err.fill(results_incl['errors']['tan'])
            c0_err.fill(results_incl['errors']['c0'])

        # Loop over wavelength and fit the zfit function
        for ilambda in range(0, data.shape[2] - lambda_step + 1, lambda_step):
            time_data = data[planet_id,:,ilambda:ilambda + lambda_step,:].sum(axis=2).mean(axis=1)
            sec_result = fit_transit(time_data, new_params=new_params, fix_params=fix_params)

            # Calculate the actual target value
            # We approximate it as area under polynom minus one, and divide by 2*width. This is kind of extrapolation to
            # the whole range that is an estimate of the transit depth due to the fitted data is normalized
            pol_pars = {par:sec_result['values'][par] for par in ['c0', 'c1', 'c2', 'c3']}
            x_points = np.linspace(0,1, 187)
            pol = polynomial(x_points, **pol_pars)
            depth_approx = (pol.sum()-1)/sec_result['values']['width']/2

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[2]:
                    # Store the best fit values
                    frac[ilambda + i] = depth_approx
                    c0[ilambda + i] = sec_result['values']['c0']
                    if not fix_params:
                        center[ilambda + i] = sec_result['values']['center']
                        width[ilambda + i] = sec_result['values']['width']
                        tan[ilambda + i] = sec_result['values']['tan']

                    # Store the parameter errors
                    frac_err[ilambda + i] = sec_result['errors']['frac']
                    c0_err[ilambda + i] = sec_result['errors']['c0']
                    if not fix_params:
                        center_err[ilambda + i] = sec_result['errors']['center']
                        width_err[ilambda + i] = sec_result['errors']['width']
                        tan_err[ilambda + i] = sec_result['errors']['tan']
                    

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[2] % lambda_step
        if remaining_bins > 0:
            # Average the data for the remaining bins
            lambdas = (data.shape[2]-remaining_bins, data.shape[2])
            time_data = data[planet_id,:,lambdas[0]:lambdas[1],:].sum(axis=2).mean(axis=1)
            sec_result = fit_transit(time_data, new_params=new_params, fix_params=fix_params)

            # Calculate the actual target value
            # We approximate it as area under polynom minus one, and divide by 2*width. This is kind of extrapolation to
            # the whole range that is an estimate of the transit depth due to the fitted data is normalized
            pol_pars = {par:sec_result['values'][par] for par in ['c0', 'c1', 'c2', 'c3']}
            x_points = np.linspace(0,1, 187)
            pol = polynomial(x_points, **pol_pars)
            depth_approx = (pol.sum()-1)/sec_result['values']['width']/2

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                # Store the best fit values
                frac[-remaining_bins + i] = depth_approx
                c0[-remaining_bins + i] = sec_result['values']['c0']
                if not fix_params:
                    center[-remaining_bins + i] = sec_result['values']['center']
                    width[-remaining_bins + i] = sec_result['values']['width']
                    tan[-remaining_bins + i] = sec_result['values']['tan']

                # Store the parameter errors
                frac_err[-remaining_bins + i] = sec_result['errors']['frac']
                c0_err[-remaining_bins + i] = sec_result['errors']['c0']
                if not fix_params:
                    center_err[-remaining_bins + i] = sec_result['errors']['frac']
                    width_err[-remaining_bins + i] = sec_result['errors']['width']
                    tan_err[-remaining_bins + i] = sec_result['errors']['tan']
                
        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(frac), np.array(center), np.array(width), np.array(tan), np.array(c0), np.array(frac_err), np.array(center_err), np.array(width_err), np.array(tan_err), np.array(c0_err), sec_result
    