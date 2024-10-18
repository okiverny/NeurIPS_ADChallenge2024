from signal_extraction_strategy import SignalExtractStrategy
import numpy as np
import warnings
from iminuit import Minuit, cost
from helpers import chi2_interpol_func, quadratic_func, find_uncertainty

# Concrete Strategies
class TemplateFitStrategy(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, planet_id, lambda_step=1, sigma_fix=None, results_incl=None):
        """
        Method to extract the stellar flux darkening due to exoplanet transition.
        Allows binning the data in wavelength bins according to lambda_step.
        
        Parameters:
        - data: the data to be fitted (time x wavelength)
        - t_min, t_max: the time range for fitting
        - lambda_step: how many wavelength bins to combine together during fitting
        - sigma_fix: =transit_breakpoint in this method
        - results_incl: results from wavelength-inclusive fit if needed
        
        Returns:
        - Arrays of signal scaling parameter: mu and its uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)
        transit_breakpoint = sigma_fix # =transit_breakpoint in this method
        
        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        total_time = data.shape[1]
        transit_half_duration = 8 # data.shape[1] // 23 # =8 time window when the stellar flux is on the half-way darkening
        ingress_time_step = transit_breakpoint - transit_half_duration
        egress_time_step = total_time - transit_breakpoint + transit_half_duration

        # Define unobscured and obscured time steps
        time_steps_unobscured_left = np.arange(0, ingress_time_step)
        time_steps_unobscured_right = np.arange(egress_time_step, total_time)
        time_steps_obscured = np.arange(ingress_time_step+2*transit_half_duration, egress_time_step-2*transit_half_duration) # avoiding edge transitions
        x = np.concatenate((time_steps_unobscured_left, time_steps_obscured, time_steps_unobscured_right))

        # mu values to create MC templates
        mu_values = np.linspace(0,0.015, 100)
        if results_incl:
            mu_incl = results_incl['mu']
            mu_values = np.linspace(mu_incl-0.002,mu_incl+0.002, 100)

        # Initialize output values
        mu_best = np.zeros(data.shape[2])
        mu_best_err = np.zeros(data.shape[2])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:
            mu_best.fill(results_incl['mu'])
            mu_best_err.fill(results_incl['mu_err'])

        # Initial parameters for the fits
        initial_params = {'a': 50, 'b':-20000, 'c':3.83e+08}
        init_pars_interpol = {'a': 1.0, 'b':0.002, 'c':0.01}


        # Loop over wavelength and fit the Erf function to normalized data, using lambda_step
        for ilambda in range(0, data.shape[2] - lambda_step + 1, lambda_step):
            # Average the data over the wavelength step
            y = data[planet_id, t_min:t_max, ilambda:ilambda + lambda_step, :].sum(axis=2).mean(axis=1)
            y_unobscured_left, y_unobscured_right = y[0:ingress_time_step], y[egress_time_step:total_time]
            y_obscured = y[ingress_time_step+2*transit_half_duration:egress_time_step-2*transit_half_duration]
            
            # statistical errors
            y_template_errors = np.concatenate((
                np.sqrt(y_unobscured_left),
                np.sqrt(y_unobscured_right),
                np.sqrt(y_obscured)))

            # Create and test the templates
            chi2_values = []
            for mu in mu_values:
                y_template = np.concatenate((y_unobscured_left, (1+mu)*y_obscured, y_unobscured_right))
                cost_function = cost.LeastSquares(x, y_template, y_template_errors, quadratic_func)
                m_bkg = Minuit(cost_function,  **initial_params)
                m_bkg.migrad()
                m_bkg.hesse()
                chi2_values.append(m_bkg.fval/(len(y_template)-len(initial_params)))

            # Fit the chi2 dependence vs mu
            chi2_values = np.array(chi2_values)
            chi2_errors = np.ones_like(chi2_values)*0.001
            try:
                cost_func = cost.LeastSquares(mu_values, chi2_values, chi2_errors, chi2_interpol_func)
                m = Minuit(cost_func,  **init_pars_interpol)
                m.migrad()
                m.hesse()
                #if not results_incl: m.visualize()

                # Estimate uncertainty from the spread at chi2_min+1
                best_mu, min_chi2 = m.values['b'], m.values['a']
                fit_func = lambda x: chi2_interpol_func(x, *m.values)
                uncertainty = find_uncertainty(fit_func, best_mu, min_chi2)
                uncertainty = m.errors['b']

                results = {'mu': m.values['b'], 'mu_err': uncertainty}
                print('Best mu=',m.values['b'], '+/-',uncertainty)

                # Check for bad fits
                if results['mu']>0.020:
                    results = {'mu': results_incl['mu'], 'mu_err': results_incl['mu_err']}

            except (RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                results = {'mu': results_incl['mu'], 'mu_err': results_incl['mu_err']} if results_incl else {'mu': 0.0005, 'mu_err': 0.0002}

            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[2]:
                    # Store the best fit values
                    mu_best[ilambda + i] = results['mu']
                    mu_best_err[ilambda + i] = results['mu_err']

        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[2] % lambda_step
        if remaining_bins > 0:
            y = data[planet_id, t_min:t_max, -remaining_bins:, :].sum(axis=2).mean(axis=1)
            y_unobscured_left, y_unobscured_right = y[0:ingress_time_step], y[egress_time_step:total_time]
            y_obscured = y[ingress_time_step+2*transit_half_duration:egress_time_step-2*transit_half_duration]
            
            # statistical errors
            y_template_errors = np.concatenate((
                np.sqrt(y_unobscured_left),
                np.sqrt(y_unobscured_right),
                np.sqrt(y_obscured)))
            
            # Create and test the templates
            chi2_values = []
            for mu in mu_values:
                y_template = np.concatenate((y_unobscured_left, (1+mu)*y_obscured, y_unobscured_right))
                cost_function = cost.LeastSquares(x, y_template, y_template_errors, quadratic_func)
                m_bkg = Minuit(cost_function,  **initial_params)
                m_bkg.migrad()
                m_bkg.hesse()
                chi2_values.append(m_bkg.fval/(len(y_template)-len(initial_params)))

            # Fit the chi2 dependence vs mu
            chi2_values = np.array(chi2_values)
            chi2_errors = np.ones_like(chi2_values)*0.001
            try:
                cost_func = cost.LeastSquares(mu_values, chi2_values, chi2_errors, chi2_interpol_func)
                m = Minuit(cost_func,  **init_pars_interpol)
                m.migrad()
                m.hesse()

                # Estimate uncertainty from the spread at chi2_min+1
                best_mu, min_chi2 = m.values['b'], m.values['a']
                fit_func = lambda x: chi2_interpol_func(x, *m.values)
                uncertainty = find_uncertainty(fit_func, best_mu, min_chi2)
                uncertainty = m.errors['b']

                results = {'mu': m.values['b'], 'mu_err': uncertainty}

                # Check for bad fits
                if results['mu']>0.015:
                    results = {'mu': results_incl['mu'], 'mu_err': results_incl['mu_err']}

            except (RuntimeError):
                print(f"Fit failed for the remaining wavelength bins. Returning inclusive values for this range.")
                results = {'mu': results_incl['mu'], 'mu_err': results_incl['mu_err']} if results_incl else {'mu': 0.0005, 'mu_err': 0.0002}

            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                mu_best[-remaining_bins + i] = results['mu']
                mu_best_err[-remaining_bins + i] = results['mu_err']
            
        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(mu_best), np.array(mu_best_err), results, None, None, None, None, None, None, None