from signal_extraction_strategy import SignalExtractStrategy
import numpy as np
import warnings
import scipy
from iminuit import Minuit, cost
from iminuit.util import make_func_code, make_with_signature, describe
from numba import vectorize, float64
from numba_stats import norm
#from minuit_helpers import model_pdf

#@vectorize([float64(float64, float64, float64, float64)], fastmath=True)
def profile_model_pdf(x, alpha, sigma, mu, muB):
    return alpha * norm.pdf(x, muB, sigma) + (1 - alpha) * norm.pdf(x, mu, sigma)

class ProfileMethod(SignalExtractStrategy):
    def parametrize_data(self, data, t_min, t_max, lambda_step=1, transit_breakpoints=None, results_incl=None):
        """
        Method for parametrization of the planet transition by double sided Erf function (simultanious fit).
        Allows binning the data in wavelength bins according to lambda_step.
        
        Parameters:
        - data: the data to be fitted (time x wavelength) after normalization/detrending
        - t_min, t_max: the time range for fitting
        - lambda_step: how many wavelength bins to combine together during fitting
        - sigma_fix: if there is a need to fix the sigma parameter in the fit (None: not fixed, float: fix value) - Does nothing here!
        - results_incl: results of wavelength-inclusive fit
        
        Returns:
        - Arrays of double gauss fit parameters: mu, sigma, alpha and their uncertainties
        """

        # Common fitting range
        t = np.arange(t_min, t_max)
        y_bias = 0.01

        # Deal with situations of ingress or egress point is within your (t_min, t_max) range
        ingress_time = transit_breakpoints[0]
        egress_time = transit_breakpoints[1]

        # Initialize output values
        mu = np.zeros(data.shape[1])
        mu_err = np.zeros(data.shape[1])
        sigma = np.zeros(data.shape[1])
        sigma_err = np.zeros(data.shape[1])
        alpha = np.zeros(data.shape[1])
        alpha_err = np.zeros(data.shape[1])

        # Initialize all values to wavelength-inclusive values from results_incl (in case of failed fits)
        if results_incl:
            mu.fill(results_incl['mu'])
            sigma.fill(results_incl['sigma'])
            alpha.fill(results_incl['alpha'])
            mu_err.fill(results_incl['mu_err'])
            sigma_err.fill(results_incl['sigma_err'])
            alpha_err.fill(results_incl['alpha_err'])

        # Initial parameters for the fit
        initial_params = {'mu': results_incl['mu'], 'sigma':results_incl['sigma'], 'alpha': results_incl['alpha']} if results_incl else {'mu': 0.003+y_bias, 'sigma': 0.0005, 'alpha': 0.6}

        # Loop over wavelength and fit the Erf function to normalized data, using lambda_step
        for ilambda in range(0, data.shape[1] - lambda_step + 1, lambda_step):
            # Add a constant bias to the transit period for better separation between out-of-transit and in-transit data
            data[ingress_time:egress_time, ilambda:ilambda + lambda_step] -= y_bias

            # Combine the data over the wavelength step
            y_data = 1 - data[t_min:t_max, ilambda:ilambda + lambda_step].T.ravel()

            try:
                c = cost.UnbinnedNLL(y_data, model_pdf)
                m = Minuit(c,  **initial_params)
                m.limits["alpha"] = (0, 1)
                m.limits["mu"] = (0, 0.5)
                m.limits["sigma"] = (0, None)
                m.migrad()

                results = {'mu': m.values['mu'], 'mu_err': m.errors['mu'],
                           'sigma': m.values['sigma'], 'sigma_err': m.errors['sigma'],
                           'alpha': m.values['alpha'], 'alpha_err': m.errors['alpha'],
                           }
                
                # Check for bad fits
                fit_convergance_status = m.fmin.is_valid
                if results['mu']>0.010 or not fit_convergance_status:
                    print(f"Fit is bad for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                    results = results_incl if results_incl else {'mu': 0.0025+y_bias, 'mu_err': 0.0020, 'sigma': 0.0005, 'sigma_err': 0.0001, 'alpha': 0.66, 'alpha_err': 0.005}
                    
            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                results = results_incl if results_incl else {'mu': 0.0025+y_bias, 'mu_err': 0.0020, 'sigma': 0.0005, 'sigma_err': 0.0001, 'alpha': 0.66, 'alpha_err': 0.005}


            # Assign the fit results to each of the individual wavelength bins within the step
            for i in range(lambda_step):
                if ilambda + i < data.shape[1]:
                    # Store the best fit values
                    mu[ilambda + i] = results['mu']
                    mu_err[ilambda + i] = results['mu_err']
                    sigma[ilambda + i] = results['sigma']
                    sigma_err[ilambda + i] = results['sigma_err']
                    alpha[ilambda + i] = results['alpha']
                    alpha_err[ilambda + i] = results['alpha_err']


        # Handle any remaining bins at the end that are less than the full lambda_step size
        remaining_bins = data.shape[1] % lambda_step
        if remaining_bins > 0:
            # Add a constant bias to the transit period for better separation between out-of-transit and in-transit data
            data[ingress_time:egress_time, -remaining_bins:] -= y_bias

            # Combine the data for the remaining bins
            y_data = 1 - data[t_min:t_max, -remaining_bins:].T.ravel()

            try:
                # Fit the double gauss function to the remaining data
                c = cost.UnbinnedNLL(y_data, model_pdf)
                m = Minuit(c,  **initial_params)
                m.limits["alpha"] = (0, 1)
                m.limits["mu"] = (0, 0.5)
                m.limits["sigma"] = (0, None)
                m.fixed["alpha"] = True
                m.migrad()
                #plt.subplots(figsize=(10, 5))
                #m.visualize()

                results = {'mu': m.values['mu'], 'mu_err': m.errors['mu'],
                           'sigma': m.values['sigma'], 'sigma_err': m.errors['sigma'],
                           'alpha': m.values['alpha'], 'alpha_err': m.errors['alpha'],
                           }
                
                # Check for bad fits
                fit_convergance_status = m.fmin.is_valid
                if results['mu']>0.010 or not fit_convergance_status:
                    print("Fit is bad for the remaining wavelength bins. Returning inclusive values for this range.")
                    results = results_incl if results_incl else {'mu': 0.0025+y_bias, 'mu_err': 0.0020, 'sigma': 0.0005, 'sigma_err': 0.0001, 'alpha': 0.66, 'alpha_err': 0.005}

            except (scipy.optimize.OptimizeWarning, RuntimeError):
                print(f"Fit failed for wavelength range {ilambda}-{ilambda+lambda_step}. Returning inclusive values for this range.")
                results = results_incl if results_incl else {'mu': 0.0025+y_bias, 'mu_err': 0.0020, 'sigma': 0.0005, 'sigma_err': 0.0001, 'alpha': 0.66, 'alpha_err': 0.005}


            # Assign the fit results to each of the remaining wavelength bins
            for i in range(remaining_bins):
                mu[-remaining_bins + i] = results['mu']
                mu_err[-remaining_bins + i] = results['mu_err']
                sigma[-remaining_bins + i] = results['sigma']
                sigma_err[-remaining_bins + i] = results['sigma_err']
                alpha[-remaining_bins + i] = results['alpha']
                alpha_err[-remaining_bins + i] = results['alpha_err']

        # Subtract y_bias from the result
        mu = np.array(mu) - y_bias

        # Return the parameter arrays (fit values and their uncertainties)
        return np.array(mu), np.array(mu_err), np.array(sigma), np.array(sigma_err), results, None, None, None, None, None
    

def fit_data(data_normalized, t_min, t_max, lambda_step=1, transit_breakpoints=None):
    # Define some parameters
    y_bias = 0.01
    N_planet, N_lambda, _ = data_normalized.shape

    #profile_model_pdf.func_code = make_func_code(("x", "alpha", "sigma", "mu"))
    profile_model_pdf._parameters = {"x": None, "alpha": (0.05, 0.9), "sigma": (0.00001, 0.1), "mu": (0.0001, 0.5), "muB": (-0.005, 0.005)}

    # Initialize output values
    mu = np.zeros((N_planet, N_lambda))
    mu_err = np.zeros((N_planet, N_lambda))
    muB = np.zeros((N_planet, N_lambda))
    muB_err = np.zeros((N_planet, N_lambda))
    sigma = np.zeros((N_planet, N_lambda))
    sigma_err = np.zeros((N_planet, N_lambda))
    alpha = np.zeros((N_planet, N_lambda))
    alpha_err = np.zeros((N_planet, N_lambda))
    

    # Loop over wavelength and fit the Erf function to normalized data, using lambda_step
    for ilambda in range(0, N_lambda - lambda_step + 1, lambda_step):
        print(f'Processing lambda {ilambda}-{ilambda + lambda_step}')
        # Set up initial guesses for independent parameters (alpha, mu) for each planet
        initial_params = {}
        initial_params["sigma"] = 0.0005
        for p in range(N_planet):
            initial_params[f"alpha_{p}"] = 0.6
            initial_params[f"mu_{p}"] = 0.003 + y_bias
            initial_params[f"muB_{p}"] = 0.0

        # Create separate UnbinnedNLL likelihoods for each planet's data using its specific PDF
        likelihoods = []
        for p in range(data_normalized.shape[0]):
            # Rename the non-shared parameters, and create the cost functions, and 
            planet_pdf = make_with_signature(profile_model_pdf, alpha=f"alpha_{p}", mu=f"mu_{p}", muB=f"muB_{p}")

            #fig, ax = plt.subplots(figsize=(16, 5))
            #plt.plot(data_normalized[p, ilambda:ilambda + lambda_step, t_min:t_max].mean(axis=0), '.', alpha=0.3, label='data, planet_id='+str(p))
            #plt.show()

            ingress_time, egress_time = transit_breakpoints[p]
            planet_data = np.copy(data_normalized[p])
            planet_data[ilambda:ilambda + lambda_step, ingress_time:egress_time] -= y_bias
            y_data_planet = 1 - planet_data[ilambda:ilambda + lambda_step, t_min:t_max].T.ravel()

            likelihoods.append( cost.UnbinnedNLL(y_data_planet, planet_pdf) )
    
        # Sum all the likelihoods to create the combined likelihood function
        combined_likelihood = sum(likelihoods)
        combined_likelihood=likelihoods[0]

        # Initialize Minuit with the combined likelihood and initial parameter values
        m = Minuit(combined_likelihood, **initial_params)

        # Run the fit
        m.migrad()
        if not m.fmin.is_valid: print('Fit failed')
        #plt.subplots(figsize=(10, 5))
        #m.visualize()

        # Assign sigma results to all lambda bins within this step
        for p in range(N_planet):
            mu[p, ilambda:ilambda + lambda_step] = m.values[f"mu_{p}"]
            mu_err[p, ilambda:ilambda + lambda_step] = m.errors[f"mu_{p}"]
            alpha[p, ilambda:ilambda + lambda_step] = m.values[f"alpha_{p}"]
            alpha_err[p, ilambda:ilambda + lambda_step] = m.errors[f"alpha_{p}"]
            sigma[p, ilambda:ilambda + lambda_step] = m.values["sigma"]
            sigma_err[p, ilambda:ilambda + lambda_step] = m.errors["sigma"]
            muB[p, ilambda:ilambda + lambda_step] = m.values[f"muB_{p}"]
            muB_err[p, ilambda:ilambda + lambda_step] = m.errors[f"muB_{p}"]


    # Remaining bins
    remaining_bins = N_lambda % lambda_step
    if remaining_bins > 0:
        # Set up initial guesses for independent parameters (alpha, mu) for each planet
        initial_params = {}
        initial_params["sigma"] = 0.001
        for p in range(N_planet):
            initial_params[f"alpha_{p}"] = 0.6
            initial_params[f"mu_{p}"] = 0.003 + y_bias
            initial_params[f"muB_{p}"] = 0.0

        # Create separate UnbinnedNLL likelihoods for each planet's data using its specific PDF
        likelihoods = []
        for p in range(data_normalized.shape[0]):
            # Rename the non-shared parameters, and create the cost functions, and 
            planet_pdf = make_with_signature(profile_model_pdf, alpha=f"alpha_{p}", mu=f"mu_{p}", muB=f"muB_{p}")

            ingress_time, egress_time = transit_breakpoints[p]
            planet_data = np.copy(data_normalized[p])
            planet_data[-remaining_bins:, ingress_time:egress_time] -= y_bias
            y_data_planet = 1 - planet_data[-remaining_bins:, t_min:t_max].T.ravel()

            likelihoods.append( cost.UnbinnedNLL(y_data_planet, planet_pdf) )
    
        # Sum all the likelihoods to create the combined likelihood function
        combined_likelihood = sum(likelihoods)
        combined_likelihood=likelihoods[0]

        # Initialize Minuit with the combined likelihood and initial parameter values
        m = Minuit(combined_likelihood, **initial_params)

        # Run the fit
        m.migrad()
        #plt.subplots(figsize=(10, 5))
        #m.visualize()

        # Assign sigma results to all lambda bins within this step
        for p in range(N_planet):
            mu[p, -remaining_bins:] = m.values[f"mu_{p}"]
            mu_err[p, -remaining_bins:] = m.errors[f"mu_{p}"]
            alpha[p, -remaining_bins:] = m.values[f"alpha_{p}"]
            alpha_err[p, -remaining_bins:] = m.errors[f"alpha_{p}"]
            sigma[p, -remaining_bins:] = m.values["sigma"]
            sigma_err[p, -remaining_bins:] = m.errors["sigma"]
            muB[p, -remaining_bins:] = m.values[f"muB_{p}"]
            muB_err[p, -remaining_bins:] = m.errors[f"muB_{p}"]


        # Subtract y_bias from the result
        mu = np.array(mu - muB) - y_bias

        # Return the parameter arrays (fit values and their uncertainties)
        return mu, mu_err, sigma, sigma_err, alpha, alpha_err, muB, muB_err