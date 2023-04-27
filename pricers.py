import numpy as np
from scipy.stats.mstats import gmean
from abc import abstractmethod

class OptionPricer:

    def __init__(self, S: np.array):
        self.S = S


    @abstractmethod
    def get_value(self):
        pass

class AsianOptionPricer(OptionPricer):

    def __init__(self, option_type: str = 'call', strike: str = 'fixed', averaging: str = 'continuous',
                 average='arithmetic', *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.strike = strike
        self.type = option_type
        self.averaging = averaging
        self.average = average


    def _get_discrete_average(self, reset, avg):
        if self.S.shape[1] % reset == 0:
            S_monitoring = np.array([self.S[:, i + self.reset] for i in range(self.S.shape[1] - reset)])
            if avg == 'arithmetic':
                average = S_monitoring.mean(axis=0)
            elif avg == 'geometric':
                average = gmean(S_monitoring, axis=0)
        else:
            raise UserWarning('Specify timesteps such that it is a multiple of reset times for discrete averaging')
        return average


    def get_value(self, r, T, K = None, reset = 12):
        if self.average == 'arithmetic':
            if self.averaging == 'continuous':
                average = self.S.mean(axis=1)
            elif self.averaging == 'discrete':
                self.reset = reset
                average = self._get_discrete_average(reset, self.average)

        elif self.average == 'geometric':
            if self.averaging == 'continuous':
                average = gmean(self.S, axis=1)
            elif self.averaging == 'discrete':
                self.reset = reset
                average = self._get_discrete_average(reset, self.average)


        n_sims = self.S.shape[0]
        call_payoff = np.zeros(n_sims)
        if self.strike == 'fixed':
            if K is None:
                raise AttributeError('User must provide strike price K for a fixed-style Asian')
            else:
                self.K = K
                call_payoff = np.mean(np.maximum(0, average-self.K))
                put_payoff = np.mean(np.maximum(0, self.K-average))
        elif self.strike == 'float':
            call_payoff = np.mean(np.maximum(0, self.S[:, -1] - average))
            put_payoff = np.mean(np.maximum(0, average - self.S[:, -1]))
        else:
            raise AttributeError('User must provide the style of the Asian option')


        call_value = np.exp(-r*T) * call_payoff
        put_value = np.exp(-r*T) * put_payoff
        return call_value, put_value

