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
reset = 12
averaging = 'discrete'
average = 'geometric'
fig = plt.figure(figsize=(11,8))

if averaging == 'discrete':
    scheme = 'Closed-Form'
    nsteps = reset
else:
    scheme = 'Euler-Maruyama'
cnt = 0
call_prices = []
put_prices = []

for n_sims in np.arange(2000, 150000, 1000):
    print(f'Number of steps: {n_sims}')
    params = {
        "S0": S0,
        "T": 1,
        "sigma": sigma,
        "rf": r,
        "n_steps": nsteps,
        "n_sims": n_sims,
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
        'strike': 'fixed',
        'S0': S0,
        'averaging': 'discrete',
        'average': 'geometric',
        'reset': 12,
        'K': K # and it will be ATM option
    }

    pricer = AsianOptionPricer(**option_params)
    call_price, put_price = pricer.get_value_mc(S=model.S)
    if cnt == 0:
        call_price_cf, put_price_cf = pricer.get_value()




    option_params = {
        'r': r,
        'T': T,
        'sigma': sigma,
        'strike': 'fixed',
        'S0': S0,
        'averaging': 'continuous',
        'average': 'geometric',
        'reset': 12,
        'K': K # and it will be ATM option
    }

    pricer = AsianOptionPricer(**option_params)
    call_price, put_price = pricer.get_value_mc(S=model.S)

    call_prices.append(call_price)
    put_prices.append(put_price)

    cnt += 1


plt.figure(figsize=(11,8))
plt.plot(call_prices, label='call prices MC', marker='x')
# plt.plot(put_prices, label='put prices MC', marker='x')
plt.axhline(call_price_cf)

plt.legend()
plt.grid(True)
plt.savefig(fr'plots\convergence_asian_{option_params["average"]}_{option_params["averaging"]}.png')
plt.show()
