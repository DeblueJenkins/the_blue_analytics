import numpy as np
from scipy.stats import norm

# Option parameters
S0 = 100    # initial stock price
K = 100     # strike price
r = 0.05    # risk-free rate
sigma = 0.2 # volatility
T = 1       # time to maturity
N = 30      # number of monitoring points
t = np.linspace(0, T, N+1) # monitoring times

# Monte Carlo parameters
M = 100000  # number of simulations
dt = T / N  # time step
Z = np.random.normal(size=(M, N)) # random normal variables for the simulations

# Discrete monitoring
S = np.zeros((M, N+1))
S[:, 0] = S0
for i in range(1, N+1):
    S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i-1])
A = np.mean(S[:, 1:], axis=1)
payoff = np.maximum(A - K, 0)
price_discrete = np.exp(-r * T) * np.mean(payoff)

# Continuous monitoring
S = np.zeros((M, N+1))
S[:, 0] = S0
for i in range(1, N+1):
    S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i-1])
A = np.mean(S[:, 1:], axis=1)
B = np.cumsum(S[:, :-1], axis=1) / np.arange(1, N+1)
V = np.maximum(A - K, 0) * np.exp(-r * t[1:])
for i in range(N-1, -1, -1):
    d1 = (np.log(B[:, i]/S[:, i]) + (r + 0.5*sigma**2)*(T-t[i])) / (sigma*np.sqrt(T-t[i]))
    d2 = d1 - sigma*np.sqrt(T-t[i])
    V = np.where(A > B[:, i], V, np.exp(-r*(t[i]-t[i+1]))*(norm.cdf(d1)*S[:, i] - norm.cdf(d2)*B[:, i]))
price_continuous = np.mean(V)

print('Discrete monitoring price:', price_discrete)
print('Continuous monitoring price:', price_continuous)