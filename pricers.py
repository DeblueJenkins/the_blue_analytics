import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import norm
from abc import abstractmethod
import warnings

class OptionPricer:

    def __init__(self, sigma: float, S0: float, r: float, T: float, K: float = None):
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T



    @abstractmethod
    def get_value(self):
        pass

class AsianOptionPricer(OptionPricer):

    def __init__(self, option_type: str = 'call', strike: str = 'fixed', averaging: str = 'continuous',
                 average='arithmetic', reset=12, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.strike = strike
        self.type = option_type
        self.averaging = averaging
        self.average = average
        self.reset = reset


    def _get_discrete_average(self, S, reset, avg):
        warnings.warn('This method is imprecise if reset!=ntimesteps in simulation. It needs to be fixed.')
        if S.shape[1] % reset == 0:

            sampler = int(S.shape[1] / reset)
            # S_monitoring = np.zeros((S.shape[0], reset))
            # S_monitoring[:, 1:] = np.array([S[:, i * sampler] for i in np.arange(1, reset)]).T
            # S_monitoring[:, 0] = S[:, 0]
            S_monitoring = np.array([S[:, i * sampler] for i in np.arange(reset)]).T

            if avg == 'arithmetic':
                average = S_monitoring.mean(axis=1)
            elif avg == 'geometric':
                average = np.power(np.product(S_monitoring, axis=1), 1/reset)
        else:
            raise UserWarning('Specify timesteps such that it is a multiple of reset times for discrete averaging')
        return average


    def get_value(self, q = 0, t1 = 0):

        b = self.r - q
        """
        :param b: cost of carry, optional, if None then risk-free rate is the only cost of carry 
        :return: (float, float): call and option value 
        """
        if self.averaging == 'continuous':

            if self.average == 'geometric':

                print("Kemna and Vorst (1990)")
                b_a = 0.5 * (b - self.sigma ** 2 / 6)
                sigma_a = self.sigma / np.sqrt(3)
                d1 = np.log(self.S0 / self.K) + (b_a + sigma_a ** 2 / 2) * self.T
                d1 = d1 / (sigma_a * np.sqrt(self.T))
                d2 = d1 - sigma_a * np.sqrt(self.T)
                call_value = self.S0 * np.exp(b_a - self.r) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                put_value = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(b_a - self.r) * norm.cdf(-d1)

            else:

                raise UserWarning('Not implemented!')


        elif self.averaging == 'discrete':

                if self.average == 'geometric':

                    warning_str = 'This is a closed-form solution, but sigma is assumed constant - not piecewise.'
                    warning_str += 'For more details, check Complete Guide for Option Pricing (Espeen Haag)'
                    warnings.warn(warning_str)
                    avrg_time = (self.T - t1)
                    var_G = pow(self.sigma, 2) * (t1 + avrg_time * (2 * self.reset - 1) / (6 * self.reset))
                    b = var_G / 2 + (self.r - q - pow(self.sigma, 2) / 2) * (t1 + avrg_time / 2)
                    d1 = (np.log(self.S0 / self.K) + b + var_G/2) / np.sqrt(var_G)
                    d2 = d1 - np.sqrt(var_G)

                    call_value = np.exp(-self.r * self.T) * (self.S0 * np.exp(b) * norm.cdf(d1) - self.K * norm.cdf(d2))
                    put_value = np.exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - self.S0 * np.exp(b) * norm.cdf(-d1))

                elif self.average == 'arithmetic':

                    warnings.warn('This is an approximation for discrete arithmetic average (Turnbull and Wakemane (1991))')

                    # this is a simplification when valuation is only required at start of contract not inside the averaging period

                    M1 = (np.exp(b*self.T) - 1) / b * self.T
                    M2_first_term = (2 * np.exp((2 * b + self.sigma ** 2)*self.T)) / ((b + self.sigma ** 2) * (2 * b + self.sigma ** 2) * self.T ** 2)
                    M2_second_term = 2 / (b * self.T ** 2)
                    M2_third_term = 1 / ((2 * b) + self.sigma ** 2) - (np.exp(b * self.T)) / (b + self.sigma ** 2)
                    M2 = M2_first_term + M2_second_term * M2_third_term
                    # adjusted cost of carry
                    b_a = np.log(M1) / self.T
                    sigma_a = np.sqrt(np.log(M2) / self.T - 2 * b_a)
                    d1 = (np.log(self.S0 / self.K) + (b_a + 0.5 * sigma_a ** 2) * self.T) / (sigma_a * np.sqrt(self.T))
                    d2 = d1 - sigma_a * np.sqrt(self.T)
                    call_value = self.S0 * np.exp((b_a - self.r) * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                    put_value = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp((b_a - self.r) * self.T) * norm.cdf(-d1)

                else:
                    raise UserWarning('When averaging is discrete, average can either be geometric or arithmetic')
        else:
            raise UserWarning('Averaging can only be "geometric" or "discrete"')

        return call_value, put_value

    def get_value_mc(self, S: np.ndarray):

        if self.average == 'arithmetic':
            average = S.mean(axis=1)
            # if self.averaging == 'continuous':
            #     average = S.mean(axis=1)
            # elif self.averaging == 'discrete':
            #     self.reset = self.reset
            #     average = self._get_discrete_average(S, self.reset, self.average)

        elif self.average == 'geometric':
            average = np.power(np.product(S, axis=1), 1/self.reset)
            # if self.averaging == 'continuous':
            #     average = gmean(S, axis=1)
            # elif self.averaging == 'discrete':
            #     self.reset = self.reset
            #     average = self._get_discrete_average(S, self.reset, self.average)


        if self.strike == 'fixed':
            if self.K is None:
                raise EOFError('User must provide strike price K for a fixed-style Asian')
            else:
                call_payoff = np.mean(np.maximum(0, average-self.K))
                put_payoff = np.mean(np.maximum(0, self.K-average))
        elif self.strike == 'float':
            call_payoff = np.mean(np.maximum(0, S[:, -1] - average))
            put_payoff = np.mean(np.maximum(0, average - S[:, -1]))
        else:
            raise EOFError('User must provide the style of the Asian option')


        call_value = np.exp(-self.r*self.T) * call_payoff
        put_value = np.exp(-self.r*self.T) * put_payoff

        return call_value, put_value

