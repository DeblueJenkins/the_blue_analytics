from simulator import GeometricBrownianMotionSimulator
from pricers import AsianOptionPricer
import matplotlib.pyplot as plt


S0 = 100
sigma = 0.2
r = 0.05
T = 1
fig = plt.figure(figsize=(11,8))

params = {
    "S0": S0,
    "T": 1,
    "sigma": sigma,
    "rf": r,
    "n_steps": 12,
    "n_sims": 30000,
    "seed": 2023,
    "scheme": 'Closed-Form'
}

model = GeometricBrownianMotionSimulator(**params)
model.simulate()
model.get_theoretical_moments()

K = 100
option_params = {
    'r': r,
    'T': T,
    'sigma': sigma,
    'strike': 'fixed',
    'S0': S0,
    'averaging': 'discrete',
    'average': 'geometric',
    'reset': 12,
    'K': K # and it will be ATM option
}

pricer = AsianOptionPricer(**option_params)
call_price, put_price = pricer.get_value_mc(S=model.S)
call_price_cf, put_price_cf = pricer.get_value()

print(f'Asian call option: {call_price:0.4f}')
print(f'Asian put option: {put_price:0.4f}')
print(f'Asian call option (closed-form): {call_price_cf:0.4f}')
print(f'Asian put option (closed-form): {put_price_cf:0.4f}')
