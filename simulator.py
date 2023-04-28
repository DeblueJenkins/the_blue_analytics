from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MonteCarloSimulator:

    def __init__(self, T: float, n_steps: float, n_sims: float, seed: int, scheme: str = None):

        self.n_steps = n_steps
        self.n_sims = n_sims
        self.seed = seed
        self.scheme = scheme
        self.T = T
        self.dt = self.T / (self.n_steps - 1)


class GeometricBrownianMotionSimulator(MonteCarloSimulator):


    def __init__(self, S0: float, sigma: float, rf: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S0 = S0
        self.sigma = sigma
        self.rf = rf
        self.get_theoretical_moments()


# these are static methods since numba/cuda will not work otherwise
    # for future development

    def get_theoretical_moments(self):
        self.theoretical_mean = self.S0 * np.exp(self.rf * self.T)
        self.theoretical_variance = self.S0 ** 2 * np.exp(2 * self.rf * self.T) * (np.exp(self.sigma ** 2 * self.T) - 1)

        # ES3 = self.S0 ** 3 * np.exp(3 * self.rf + 3 * self.sigma)
        # self.theoretical_skewness = (ES3 - 3 * self.rf * self.sigma ** 2 + 2 * self.rf ** 3) / self.sigma ** 3

    @staticmethod
    def _closed_form(S_prev, rf, dt, sigma, normal):
        pass

    @staticmethod
    def _euler_maruyama_scheme(S_prev, rf, dt, sigma, normal):
        return S_prev * (1 + rf * dt + sigma * normal * np.sqrt(dt))

    @staticmethod
    def _milstein_scheme(S_prev, rf, dt, sigma, normal):
        dW = normal * np.sqrt(dt)
        return S_prev * (1 + rf * dt + sigma * dW + 0.5 * sigma**2 * (dW ** 2 - dt))

    @staticmethod
    def _runge_kutta(S_prev, rf, dt, sigma, normal):
        dW = normal * np.sqrt(dt)
        S_hat = S_prev * (1 + rf * dt + sigma * np.sqrt(dt))
        euler_term = S_prev * (1 + rf * dt + sigma * normal * np.sqrt(dt))
        adjustment = 0.5 * (sigma * (S_hat - S_prev)) * (dW ** 2 - dt) / np.sqrt(dt)
        return euler_term + adjustment

    @staticmethod
    def _closed_form_soln(S_prev, rf, dt, sigma, normal):
        W = normal * np.sqrt(dt)
        return S_prev * np.exp((rf - sigma ** 2 / 2) * dt + sigma * W)

    @staticmethod
    def _simulate(S0, rf, dt, sigma, normal, n_sims, n_steps, func: object):

        S = np.zeros((n_sims, n_steps))
        S[:,0] = S0

        for i in range(n_steps-1):
            S[:,i+1] = func(S[:,i], rf, dt, sigma, normal[:,i])

        return S


    def simulate(self):
        np.random.seed(self.seed)
        normal = np.random.normal(0, 1, size=(self.n_sims, self.n_steps))

        if self.scheme == 'Euler-Maruyama':
            func = self._euler_maruyama_scheme
        elif self.scheme == 'Milstein':
            func = self._milstein_scheme
        elif self.scheme == 'Runge-Kutta':
            func = self._runge_kutta
        else:
            func = self._closed_form_soln

        self.S = self._simulate(self.S0, self.rf, self.dt, self.sigma, normal, self.n_sims, self.n_steps, func)


    def plot_paths(self):
        if self.scheme is not None:
            plt.title(f"Simulated GBM paths using {self.scheme} discr. scheme")
            if self.scheme is not None:
                for j in range(self.n_sims):
                    plt.plot(self.S[j,:])
                    plt.grid(True)

    def plot_dist(self):
        plt.title(f"Distribution using {self.scheme} discr. scheme")
        sns.histplot(self.S[:,-1])
        plt.show()








