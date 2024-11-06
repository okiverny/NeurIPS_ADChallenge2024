import numpy as np
import scipy
import warnings

from background_subtraction import BackgroundSubtractionStrategy
from signal_extraction_strategy import SignalExtractStrategy
from signal_extraction_strategy import ErfParametrization, DoubleSidedErfParametrization, DSErfMinuitParametrization, ZFitParametrization, MinuitParametrization
from template_fit_strategy import TemplateFitStrategy
from ProfileStrategy import ProfileMethod
from DoubleSidedErfReducedParametrization import DoubleSidedErfReducedParametrization
from helpers import optimize_breakpoint

class SignalExtractor:
    """The SignalExtractor defines the interface to BackgroundSubtractionStrategy and SignalExtractStrategy."""

    def __init__(self, background_strategy: BackgroundSubtractionStrategy, signal_extract_strategy: SignalExtractStrategy) -> None:
        self._background_strategy = background_strategy
        self._signal_strategy = signal_extract_strategy

    def background_strategy(self) -> BackgroundSubtractionStrategy:
        """Just in case we need a reference to one of BackgroundSubtractionStrategy objects"""
        return self._background_strategy

    def set_background_strategy(self, background_strategy: BackgroundSubtractionStrategy) -> None:
        """Usually, the SignalExtractor allows replacing a BackgroundSubtractionStrategy object at runtime."""
        self._background_strategy = background_strategy

    def signal_extract_strategy(self) -> SignalExtractStrategy:
        """Just in case we need a reference to one of SignalExtractStrategy objects"""
        return self._signal_strategy

    def set_signal_extract_strategy(self, signal_extract_strategy: SignalExtractStrategy) -> None:
        """Usually, the SignalExtractor allows replacing a SignalExtractStrategy object at runtime."""
        self._signal_strategy = signal_extract_strategy

    def extract_signal(self, data, data_FGS, planet_id, lambda_step=10):
        """The SignalExtractor delegates some work to the BackgroundSubtractionStrategy object."""

        # Find breakpoints (ingress/egress) for time steps using the optimize_breakpoint() method. We find the breakpoints inclusively in wavelength
        initial_breakpoint = (data.shape[1] // 3) - 2  # 60 --> tune this value
        transit_breakpoint = optimize_breakpoint(data[planet_id, :, :, :].mean(axis=(1,2)), initial_breakpoint)
        print('Optimized breakpoint is:', transit_breakpoint, ' (planet_id=', planet_id, ')')

        # Pass this breakpoints to the background strategy
        data_normalized, quality_metric, total_flux = self._background_strategy.subtract_background(
            data, planet_id, transit_breakpoint
        )
        data_normalized_FGS, quality_metric_FGS, total_flux_FGS = self._background_strategy.subtract_background(
            data_FGS, planet_id, transit_breakpoint
        )
        print(f'Background subtraction chi2/ndf quality check (FGS): {quality_metric_FGS[0]:.2f}')

        # ############ Simple Erf Functions ####################
        if isinstance(self.signal_extract_strategy(), ErfParametrization):
            # First, fit the function on data inclusive in wavelength to get parameters. We use approximate time ranges.
            results_incl_ingress = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=data_normalized.shape[1])  # 0, 115
            transit_breakpoint = int(results_incl_ingress[2][0])
            print('Improved breakpoint is:', transit_breakpoint)

            # Refitting one more time
            results_incl_ingress = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=data_normalized.shape[1])
            results_incl_egress = self._signal_strategy.parametrize_data(data_normalized, transit_breakpoint+12, data.shape[1], lambda_step=data_normalized.shape[1])

            # Check if ingress and egress time are symmetrical
            if abs(data.shape[1]-results_incl_ingress[2][0]-results_incl_egress[2][0]) > 7.0:
                print('WARNING: Ingress/egress times are not symmetrical. Please, check the code!')

            # Results from the fitting of ingress
            a_vals, c_vals, t0_vals, sigma_vals, a_errs, c_errs, t0_errs, sigma_errs = self._signal_strategy.parametrize_data(
                data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=lambda_step, results_incl=results_incl_ingress
            )
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, sigma_vals_FGS, a_errs_FGS, c_errs_FGS, t0_errs_FGS, sigma_errs_FGS = self._signal_strategy.parametrize_data(
                data_normalized_FGS, 0, data_FGS.shape[1]-transit_breakpoint-12, lambda_step=data_normalized_FGS.shape[1], results_incl=results_incl_ingress
            )

            # Results from the fitting of egress
            a_vals_eg, *_ = self._signal_strategy.parametrize_data(
                data_normalized, transit_breakpoint+12, data.shape[1], lambda_step=lambda_step, results_incl=results_incl_egress
            )
            a_vals_eg_FGS, *_ = self._signal_strategy.parametrize_data(
                data_normalized_FGS, transit_breakpoint+12, data_FGS.shape[1], lambda_step=data_normalized_FGS.shape[1], results_incl=results_incl_egress
            )

            # Combine 'a' values
            a_vals = 0.5*(np.abs(a_vals) + np.abs(a_vals_eg))
            a_vals_FGS = 0.5*(np.abs(a_vals_FGS) + np.abs(a_vals_eg_FGS))
        ############ End Simple Erf Functions ####################

        ############ Two-Sided Erf Function ####################
        if isinstance(self.signal_extract_strategy(), DoubleSidedErfParametrization) or isinstance(self.signal_extract_strategy(), DoubleSidedErfReducedParametrization):
            # First, fit the function on data inclusive in wavelength to get parameters.
            results_incl = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=data_normalized.shape[1])
            transit_breakpoints = (int(results_incl[2][0]), int(results_incl[3][0]))
            if results_incl[0][0]==0.003:
                print('Trying to find good fits among smaller steps...')
                # Run the parameterization again
                a_vals, c_vals, t0_ingress_vals, t0_egress_vals, sigma_vals, a_errs, c_errs, t0_ingress_errs, t0_egress_errs, sigma_errs = \
                    self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=15)
                
                mask = a_vals != 0.003  # Create a mask for elements not equal to 0.003
                if np.any(mask): 
                    results_incl = (np.array([a_vals[mask].mean()]), np.array([c_vals[mask].mean()]), np.array([t0_ingress_vals[mask].mean()]), np.array([t0_egress_vals[mask].mean()]), np.array([sigma_vals[mask].mean()]),
                                np.array([a_errs[mask].mean()]), np.array([c_errs[mask].mean()]), np.array([t0_ingress_errs[mask].mean()]), np.array([t0_egress_errs[mask].mean()]), np.array([sigma_errs[mask].mean()]) )
                    transit_breakpoints = (int(t0_ingress_vals[mask].mean()), int(t0_egress_vals[mask].mean()))
                
            transit_breakpoint = int(results_incl[2][0])
            print('Improved breakpoints are:', transit_breakpoints)

            # Check if ingress and egress time are symmetrical
            if abs(data.shape[1]-results_incl[2][0]-results_incl[3][0]) > 7.0:
                print('WARNING: Ingress/egress times are not symmetrical. Please, check the code!')

            # Results from the fitting of the full time range
            a_vals, c_vals, t0_vals, _, sigma_vals, a_errs, c_errs, t0_errs, _, sigma_errs = self._signal_strategy.parametrize_data(
                data_normalized, 0, data.shape[1], lambda_step=lambda_step, results_incl=results_incl
            )

            # Fitting FGS data
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, _, sigma_vals_FGS, a_errs_FGS, c_errs_FGS, t0_errs_FGS, _, sigma_errs_FGS = self._signal_strategy.parametrize_data(
                data_normalized_FGS, 0, data_FGS.shape[1], lambda_step=data_normalized_FGS.shape[1], results_incl=results_incl
            )

            # Due to Erf parametrization, the solution is 2*a_vals and need to increase the error from the fit
            a_vals, a_vals_FGS = 2*np.abs(a_vals), 2*np.abs(a_vals_FGS)
            a_errs, a_errs_FGS = 1*np.abs(a_errs), 1*np.abs(a_errs_FGS)
        ############ End Two-Sided Erf Function ####################

        ############ Two-Sided Erf Function (Minuit) ####################
        if isinstance(self.signal_extract_strategy(), DSErfMinuitParametrization):
            # First, fit the function on data inclusive in wavelength to get parameters.
            results_incl = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=data_normalized.shape[1])
            transit_breakpoint = int(results_incl[2][0])
            print('Improved breakpoint is:', transit_breakpoint)

            # Check if ingress and egress time are symmetrical
            if abs(data.shape[1]-results_incl[2][0]-results_incl[3][0]) > 7.0:
                print('WARNING: Ingress/egress times are not symmetrical. Please, check the code!')

            # Results from the fitting of the full time range
            a_vals, c_vals, t0_vals, _, sigma_vals, a_errs, c_errs, t0_errs, _, sigma_errs = self._signal_strategy.parametrize_data(
                data_normalized, 0, data.shape[1], lambda_step=lambda_step, results_incl=results_incl
            )

            # Fitting FGS data
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, _, sigma_vals_FGS, a_errs_FGS, c_errs_FGS, t0_errs_FGS, _, sigma_errs_FGS = self._signal_strategy.parametrize_data(
                data_normalized_FGS, 0, data_FGS.shape[1], lambda_step=data_normalized_FGS.shape[1], results_incl=results_incl
            )

            # Due to Erf parametrization, the solution is 2*a_vals and need to increase the error from the fit
            a_vals, a_vals_FGS = 2*np.abs(a_vals), 2*np.abs(a_vals_FGS)
            a_errs, a_errs_FGS = 3*np.abs(a_errs), 3*np.abs(a_errs_FGS)
        ############ End Two-Sided Erf Function (Minuit) ####################

        ############ ZFit Function ####################
        if isinstance(self.signal_extract_strategy(), ZFitParametrization):
            # First, fit the function on data inclusive in wavelength to get parameters.
            *_, results_incl = self._signal_strategy.parametrize_data(data, 0, data.shape[1], planet_id, lambda_step=data_normalized.shape[1])
            transit_breakpoint = int( (data.shape[1] // 2) - results_incl.params['width_0']['value']*data.shape[1])
            print('Improved breakpoint is:', transit_breakpoint)

            # Results from the fitting of the full time range
            frac, center, width, tan, c0, frac_err, center_err, width_err, tan_err, c0_err, _ = self._signal_strategy.parametrize_data(
                data, 0, data.shape[1], planet_id, lambda_step=lambda_step, results_incl=results_incl
            )
            a_vals, c_vals, t0_vals, sigma_vals = 0.5*frac/results_incl.params['width_0']['value'], center, width, tan
            a_errs, c_errs, t0_errs, sigma_errs = frac_err, center_err, width_err, tan_err

            # Fitting FGS data
            frac_FGS, center_FGS, width_FGS, tan_FGS, c0_FGS, frac_err_FGS, center_err_FGS, width_err_FGS, tan_err_FGS, c0_err_FGS, _ = self._signal_strategy.parametrize_data(
                data_FGS, 0, data.shape[1], planet_id, lambda_step=data_FGS.shape[2], results_incl=results_incl
            )
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, sigma_vals_FGS = 0.5*frac_FGS/results_incl.params['width_0']['value'], center_FGS, width_FGS, tan_FGS
            a_errs_FGS, c_errs_FGS, t0_errs_FGS, sigma_errs_FGS = frac_err_FGS, center_err_FGS, width_err_FGS, tan_err_FGS
        ############ End ZFit Function ####################

        ############ Minuit Function ####################
        if isinstance(self.signal_extract_strategy(), MinuitParametrization):
            # First, fit the function on data inclusive in wavelength to get parameters.
            *_, results_incl = self._signal_strategy.parametrize_data(data, 0, data.shape[1], planet_id, lambda_step=data_normalized.shape[1])
            transit_breakpoint = int( (data.shape[1] // 2) - results_incl['values']['width']*data.shape[1])
            print('Improved breakpoint is:', transit_breakpoint)

            # Results from the fitting of the full time range
            frac, center, width, tan, c0, frac_err, center_err, width_err, tan_err, c0_err, _ = self._signal_strategy.parametrize_data(
                data, 0, data.shape[1], planet_id, lambda_step=lambda_step, results_incl=results_incl
            )
            a_vals, c_vals, t0_vals, sigma_vals = frac, center, width, tan
            a_errs, c_errs, t0_errs, sigma_errs = frac_err, center_err, width_err, tan_err

            # Fitting FGS data
            frac_FGS, center_FGS, width_FGS, tan_FGS, c0_FGS, frac_err_FGS, center_err_FGS, width_err_FGS, tan_err_FGS, c0_err_FGS, _ = self._signal_strategy.parametrize_data(
                data_FGS, 0, data.shape[1], planet_id, lambda_step=data_FGS.shape[2], results_incl=results_incl
            )
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, sigma_vals_FGS = frac_FGS, center_FGS, width_FGS, tan_FGS
            a_errs_FGS, c_errs_FGS, t0_errs_FGS, sigma_errs_FGS = frac_err_FGS, center_err_FGS, width_err_FGS, tan_err_FGS
        ############ End Minuit Function ####################

        ############ Template Fit ####################
        if isinstance(self.signal_extract_strategy(), TemplateFitStrategy):
            # Try to improve transit boundaries with DoubeleSidedErf function
            self.set_signal_extract_strategy(DoubleSidedErfParametrization())
            results_erf_incl = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=data_normalized.shape[1])
            transit_breakpoints = (int(results_erf_incl[2][0]), int(results_erf_incl[3][0]))
            transit_breakpoint = transit_breakpoints[0]
            print('Improved breakpoints are:', transit_breakpoints)

            # Switch back to Template Fit Strategy
            self.set_signal_extract_strategy(TemplateFitStrategy())

            # First, fit the function on data inclusive in wavelength to get parameters.
            _, _, results_incl, *_ = self._signal_strategy.parametrize_data(
                data, 0, data.shape[1], planet_id, lambda_step=data_normalized.shape[1], sigma_fix=int(transit_breakpoint)
            )

            # Run the inclusive estimate one more time with a better templating around the answer
            _, _, results_incl, *_ = self._signal_strategy.parametrize_data(
                data, 0, data.shape[1], planet_id, lambda_step=data_normalized.shape[1], sigma_fix=int(transit_breakpoint), results_incl=results_incl
            )

            # Results from the fitting of the full time range
            mu, mu_err, *_ = self._signal_strategy.parametrize_data(
                data, 0, data.shape[1], planet_id, lambda_step=lambda_step, sigma_fix=int(transit_breakpoint), results_incl=results_incl
            )
            a_vals, c_vals, t0_vals, sigma_vals = np.abs(mu), mu, mu, mu
            a_errs, c_errs, t0_errs, sigma_errs = 1*mu_err, mu_err, mu_err, mu_err

            # Fitting FGS data
            mu_FGS, mu_err_FGS, *_ = self._signal_strategy.parametrize_data(
                data_FGS, 0, data.shape[1], planet_id, lambda_step=data_FGS.shape[2], sigma_fix=int(transit_breakpoint), results_incl=results_incl
            )
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, sigma_vals_FGS = np.abs(mu_FGS), mu_FGS, mu_FGS, mu_FGS
            a_errs_FGS, c_errs_FGS, t0_errs_FGS, sigma_errs_FGS = 1*mu_err_FGS, mu_err_FGS, mu_err_FGS, mu_err_FGS
        ############ End Template Fit ####################

        ############ Profile Double Gauss ####################
        if isinstance(self.signal_extract_strategy(), ProfileMethod):
            # Try to improve transit boundaries with DoubeleSidedErf function
            self.set_signal_extract_strategy(DoubleSidedErfParametrization())
            results_erf_incl = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=data_normalized.shape[1])
            transit_breakpoints = (int(results_erf_incl[2][0]), int(results_erf_incl[3][0]))
            if results_erf_incl[0][0]==0.003:
                print('Trying to find good fits among smaller steps...')
                # Run the parameterization again
                a_vals, c_vals, t0_ingress_vals, t0_egress_vals, sigma_vals, a_errs, c_errs, t0_ingress_errs, t0_egress_errs, sigma_errs = \
                    self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=15)
                
                mask = a_vals != 0.003  # Create a mask for elements not equal to 0.003
                if np.any(mask): 
                    results_erf_incl = (np.array([a_vals[mask].mean()]), np.array([c_vals[mask].mean()]), np.array([t0_ingress_vals[mask].mean()]), np.array([t0_egress_vals[mask].mean()]), np.array([sigma_vals[mask].mean()]),
                                np.array([a_errs[mask].mean()]), np.array([c_errs[mask].mean()]), np.array([t0_ingress_errs[mask].mean()]), np.array([t0_egress_errs[mask].mean()]), np.array([sigma_errs[mask].mean()]) )
                    transit_breakpoints = (int(t0_ingress_vals[mask].mean()), int(t0_egress_vals[mask].mean()))
            print('Improved breakpoints are:', transit_breakpoints)

            # Switch back to Template Fit Strategy
            self.set_signal_extract_strategy(ProfileMethod())

            # It is expected that the fit won't work on data inclusive in wavelength. Therefore we get inclusive parameters
            # from DoubleSidedErfParametrization approach
            results_incl = {'mu': 2*results_erf_incl[0][0], 'mu_err': results_erf_incl[5][0], 'sigma': 0.0005, 'sigma_err': 0.0001, 'alpha': 0.66, 'alpha_err': 0.01 }

            # Results from the fitting of the full time range
            mu, mu_err, sigma, sigma_err, *_ = self._signal_strategy.parametrize_data(
                data_normalized, 0, data.shape[1], lambda_step=lambda_step, transit_breakpoints=transit_breakpoints, results_incl=results_incl
            )
            a_vals, c_vals, t0_vals, sigma_vals = np.abs(mu), sigma, mu, mu
            a_errs, c_errs, t0_errs, sigma_errs = mu_err, sigma_err, mu_err, mu_err

            # Fitting FGS data
            mu_FGS, mu_err_FGS, sigma_FGS, sigma_err_FGS, *_ = self._signal_strategy.parametrize_data(
                data_normalized_FGS, 0, data_FGS.shape[1], lambda_step=data_FGS.shape[2], transit_breakpoints=transit_breakpoints, results_incl=results_incl
            )
            a_vals_FGS, c_vals_FGS, t0_vals_FGS, sigma_vals_FGS = np.abs(mu_FGS), sigma_FGS, mu_FGS, mu_FGS
            a_errs_FGS, c_errs_FGS, t0_errs_FGS, sigma_errs_FGS = mu_err_FGS, sigma_err_FGS, mu_err_FGS, mu_err_FGS
        ############ End Profile Double Gauss ####################

        # Combine data and data_FGS results. Reverse lambda ordering
        output = (
            np.concatenate((data_normalized_FGS.T, data_normalized.T[::-1])),
            quality_metric,
            np.concatenate((a_vals_FGS, a_vals[::-1])),
            np.concatenate((c_vals_FGS, c_vals[::-1])),
            np.concatenate((t0_vals_FGS, t0_vals[::-1])),
            np.concatenate((sigma_vals_FGS, sigma_vals[::-1])),
            np.concatenate((a_errs_FGS, a_errs[::-1])),
            np.concatenate((c_errs_FGS, c_errs[::-1])),
            np.concatenate((t0_errs_FGS, t0_errs[::-1])),
            np.concatenate((sigma_errs_FGS, sigma_errs[::-1])),
            (total_flux_FGS, total_flux),
            transit_breakpoints
            )

        return output