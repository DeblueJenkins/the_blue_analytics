import numpy as np
from scipy.stats import norm

np.random.seed(2023)
def descrete_asian_montecarlo(S0: float, T: float, sigma: float,
                              r: float, q: float, t1: float,n_sims: int, n: int, return_time: bool = False):

    dt_avrg = (T - t1)/(n-1)

    S = np.zeros((n, n_sims))
    w = np.random.standard_normal(n_sims)

    mean = (r - q - pow(sigma, 2)/2) * t1
    std = sigma * np.sqrt(t1)
    S[0, :] = S0 * np.exp(mean + std * w)
    time = t1
    times = [time]

    for i in range(1, n):
        w = np.random.standard_normal(n_sims)
        mean = (r - q - pow(sigma, 2)/2) * dt_avrg
        std = sigma * np.sqrt(dt_avrg)
        time += dt_avrg
        S[i, :] = S[i-1, :] * np.exp(mean + std * w)
        times.append(time)

    if return_time:
        return S, times
    else:
        return S

def geom_asian_payoff(S, E: float, r: float, T: float, n: int):
    A = np.power(np.product(S, axis = 0), 1/n)
    return np.exp(-r*T) * np.maximum(0, A - E)

def geometric_asian(S0: float, T: float, E: float, sigma: float, r: float, q: float, t1: float, n: int):

    avrgTime = (T - t1)
    var_G = pow(sigma, 2) * (t1 + avrgTime * (2 * n - 1) / (6 * n))
    b = var_G / 2 + (r - q - pow(sigma, 2) / 2) * (t1 + avrgTime / 2)
    d1 = (np.log(S0 / E) + b + var_G/2) / np.sqrt(var_G)
    d2 = d1 - np.sqrt(var_G)

    return np.exp(-r * T) * (S0 * np.exp(b) * norm.cdf(d1) - E * norm.cdf(d2))

S_start_0 = descrete_asian_montecarlo(S0=100, T=1.0, sigma=0.2, r=0.05, q=0.0, t1=0.0, n_sims=30000, n=12)
V_start_0 = geom_asian_payoff(S=S_start_0, E=100, r=0.05, T=1.0, n=12)
V_cf_0 = geometric_asian(S0=100, T=1, E=100, sigma=0.2, r=0.05, q=0, t1=0, n=12)
print(np.mean(V_start_0))
print(V_cf_0)