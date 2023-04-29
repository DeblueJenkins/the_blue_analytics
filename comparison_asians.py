from simulator import GeometricBrownianMotionSimulator
from pricers import AsianOptionPricer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

S0 = 100
sigma = 0.2
r = 0.05
T = 1
nsteps = 1000
nsims = 30000
reset = 12
averaging = 'continuous'
average = 'geometric'
fig = plt.figure(figsize=(11,8))

if averaging == 'discrete':
    scheme = 'Closed-Form'
    nsteps = reset
else:
    scheme = 'Euler-Maruyama'

plot_strikes_range = False

params = {
    "S0": S0,
    "T": 1,
    "sigma": sigma,
    "rf": r,
    "n_steps": nsteps,
    "n_sims": nsims,
    "seed": 2023,
    "scheme": 'Closed-Form'
}

model = GeometricBrownianMotionSimulator(**params)
model.simulate()
model.get_theoretical_moments()

K = 100
Ks = np.linspace(0.8, 1.2)
# Ts = np.arange(0.5, 1, 1.5)
# table = pd.DataFrame(columns=Ks, index=Ts)
#

table = pd.DataFrame(index=['call', 'put'], columns=['approx/closed', 'monte carlo'])

option_params = {
    'r': r,
    'T': T,
    'sigma': sigma,
    'strike': 'float',
    'S0': S0,
    'averaging': averaging,
    'average': average,
    'reset': reset,
    'K': K # and it will be ATM option
}

pricer = AsianOptionPricer(**option_params)
call_price, put_price = pricer.get_value_mc(S=model.S)
call_price_cf, put_price_cf = pricer.get_value()

table.loc['call', 'approx/closed'] = call_price_cf
table.loc['call', 'monte carlo'] = call_price
table.loc['put', 'approx/closed'] = put_price_cf
table.loc['put', 'monte carlo'] = put_price

print(table.to_latex(float_format="%.4f"))

print(f'Asian call option: {call_price:0.4f}')
print(f'Asian put option: {put_price:0.4f}')
print(f'Asian call option (closed-form): {call_price_cf:0.4f}')
print(f'Asian put option (closed-form): {put_price_cf:0.4f}')

call_prices = []
put_prices = []
call_prices_cf = []
put_prices_cf = []

if plot_strikes_range:

    for k in Ks:
        K = S0 * k



        option_params = {
            'r': r,
            'T': T,
            'sigma': sigma,
            'strike': 'fixed',
            'S0': S0,
            'averaging': averaging,
            'average': average,
            'reset': 12,
            'K': K # and it will be ATM option
        }

        pricer = AsianOptionPricer(**option_params)
        call_price, put_price = pricer.get_value_mc(S=model.S)
        call_price_cf, put_price_cf = pricer.get_value()

        call_prices.append(call_price)
        put_prices.append(put_price)
        call_prices_cf.append(call_price_cf)
        put_prices_cf.append(put_price_cf)

    plt.figure(figsize=(11,8))
    plt.scatter(Ks * S0, call_prices, label='call prices MC', marker='x')
    plt.scatter(Ks * S0, put_prices, label='put prices MC', marker='x')
    plt.scatter(Ks * S0, call_prices_cf, label='call prices approx.')
    plt.scatter(Ks * S0, put_prices_cf, label='put prices approx.')
    plt.legend()
    plt.grid(True)
    plt.savefig(fr'plots\asian_{option_params["average"]}_{option_params["averaging"]}.png')
    plt.show()


