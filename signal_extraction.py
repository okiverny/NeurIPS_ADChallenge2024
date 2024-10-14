from background_subtraction import BackgroundSubtractionStrategy
from signal_extraction_strategy import SignalExtractStrategy
from helpers import optimize_breakpoint
import numpy as np
import scipy
import warnings

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

    def set_signal_extract_strategy(self, signal_extract_strategy: SignalExtractStrategy) -> None:
        """Usually, the SignalExtractor allows replacing a SignalExtractStrategy object at runtime."""
        self._signal_strategy = signal_extract_strategy

    def extract_signal(self, data, data_FGS, planet_id):
        """The SignalExtractor delegates some work to the BackgroundSubtractionStrategy object."""

        # Find breakpoints (ingress/egress) for time steps using the optimize_breakpoint() method. We find the breakpoints inclusively in wavelength
        initial_breakpoint = (data.shape[1] // 3) - 2  # 60 --> tune this value
        transit_breakpoint = optimize_breakpoint(data[planet_id, :, :, :].mean(axis=(1,2)), initial_breakpoint)
        print('Optimized breakpoint is:', transit_breakpoint, ' (planet_id=', planet_id, ')')

        # Pass this breakpoints to the background strategy
        data_normalized, quality_metric = self._background_strategy.subtract_background(
            data, planet_id, transit_breakpoint
        )
        data_normalized_FGS, _ = self._background_strategy.subtract_background(
            data_FGS, planet_id, transit_breakpoint
        )

        # ############ Simple Erf Functions ####################
        # # First, fit the function on data inclusive in wavelength to get parameters. We use approximate time ranges.
        # results_incl_ingress = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=data_normalized.shape[1])  # 0, 115
        # transit_breakpoint = int(results_incl_ingress[2][0])
        # print('Improved breakpoint is:', transit_breakpoint)

        # # Refitting one more time
        # results_incl_ingress = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=data_normalized.shape[1])
        # results_incl_egress = self._signal_strategy.parametrize_data(data_normalized, transit_breakpoint+12, data.shape[1], lambda_step=data_normalized.shape[1])

        # # Check if ingress and egress time are symmetrical
        # if abs(data.shape[1]-results_incl_ingress[2][0]-results_incl_egress[2][0]) > 7.0:
        #     print('WARNING: Ingress/egress times are not symmetrical. Please, check the code!')


        # # Results from the fitting of ingress
        # a_vals, c_vals, t0_vals, sigma_vals, a_errs, c_errs, t0_errs, sigma_errs = self._signal_strategy.parametrize_data(
        #     data_normalized, 0, data.shape[1]-transit_breakpoint-12, lambda_step=12, results_incl=results_incl_ingress
        # )

        # # Results from the fitting of egress
        # a_vals_eg, c_vals_eg, t0_vals_eg, sigma_vals_eg, a_errs_eg, c_errs_eg, t0_errs_eg, sigma_errs_eg = self._signal_strategy.parametrize_data(
        #     data_normalized, transit_breakpoint+12, data.shape[1], lambda_step=12, results_incl=results_incl_egress
        # )

        # # Combine 'a' values
        # a_vals = 0.5*(np.abs(a_vals) + np.abs(a_vals_eg))
        # ############ End Simple Erf Functions ####################


        ############ Two-Sided Erf Functions ####################
        # First, fit the function on data inclusive in wavelength to get parameters.
        results_incl = self._signal_strategy.parametrize_data(data_normalized, 0, data.shape[1], lambda_step=data_normalized.shape[1])
        transit_breakpoint = int(results_incl[2][0])
        print('Improved breakpoint is:', transit_breakpoint)

        # Check if ingress and egress time are symmetrical
        if abs(data.shape[1]-results_incl[2][0]-results_incl[3][0]) > 7.0:
            print('WARNING: Ingress/egress times are not symmetrical. Please, check the code!')

        # Results from the fitting of the full time range
        a_vals, c_vals, t0_vals, _, sigma_vals, a_errs, c_errs, t0_errs, _, sigma_errs = self._signal_strategy.parametrize_data(
            data_normalized, 0, data.shape[1], lambda_step=13, results_incl=results_incl
        )

        # Fitting FGS data
        a_vals_FGS, c_vals_FGS, t0_vals_FGS, _, sigma_vals_FGS, a_errs_FGS, c_errs_FGS, t0_errs_FGS, _, sigma_errs_FGS = self._signal_strategy.parametrize_data(
            data_normalized_FGS, 0, data_FGS.shape[1], lambda_step=data_normalized_FGS.shape[1], results_incl=results_incl
        )
        ############ End Two-Sided Erf Functions ####################

        # Combine data and data_FGS results. Reverse lambda ordering
        a_vals = np.concatenate((a_vals_FGS, a_vals[::-1]))
        c_vals = np.concatenate((c_vals_FGS, c_vals[::-1]))
        t0_vals = np.concatenate((t0_vals_FGS, t0_vals[::-1]))
        sigma_vals = np.concatenate((sigma_vals_FGS, sigma_vals[::-1]))
        a_errs = np.concatenate((a_errs_FGS, a_errs[::-1]))
        c_errs = np.concatenate((c_errs_FGS, c_errs[::-1]))
        t0_errs = np.concatenate((t0_errs_FGS, t0_errs[::-1]))
        sigma_errs = np.concatenate((sigma_errs_FGS, sigma_errs[::-1]))

        return data_normalized, quality_metric, a_vals, c_vals, t0_vals, sigma_vals, a_errs, c_errs, t0_errs, sigma_errs