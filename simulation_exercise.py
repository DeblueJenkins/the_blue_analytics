from simulator import GeometricBrownianMotionSimulator
from scipy.stats import describe, skew
import numpy as np
from pricers import AsianOptionPricer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import time


def main(run='simulation_exercise'):
    print(run)

    if run == 'simulation_exercise':

        fig = plt.figure(figsize=(11,8))

        schemes = ['Closed-form', 'Euler-Maruyama', 'Runge-Kutta', 'Milstein']
        table = pd.DataFrame(index=schemes, columns=['var error', 'mean error'])
        for scheme in schemes:


            print(f'scheme: {scheme}')

            params = {
                "S0": 100,
                "T": 1,
                "sigma": 0.2,
                "rf": 0.05,
                "n_steps": 1000,
                "n_sims": 100000,
                "seed": 2023,
                "scheme": scheme
            }

            # print(params)

            if scheme == 'Closed-form':
                params['scheme'] = None


            model = GeometricBrownianMotionSimulator(**params)
            t0 = time.time()
            model.simulate()
            print(np.round(time.time() - t0, 3))
            # model.plot_paths()
            # model.plot_dist()

            model.get_theoretical_moments()

            res = describe(model.S[:,-1])

            plt.hist(model.S[:,-1], bins=200, alpha=0.7, label=scheme)


            # print(res)
            var_error = np.round(np.abs(100 * (res.variance - model.theoretical_variance) / model.theoretical_variance), 4)
            mean_error = np.round(np.abs(100 * (res.mean - model.theoretical_mean) / model.theoretical_mean), 4)
            # skew_error = np.round(np.abs(100 * (res.skewness - model.theoretical_skewness) / model.theoretical_skewness), 4)

            table.loc[scheme, 'var error'] = var_error
            table.loc[scheme, 'mean error'] = mean_error
            # table.loc[scheme, 'skew error'] = skew_error

        plt.legend()
        plt.grid()
        plt.savefig(r'plots\distplots_gbm_sims.png')
        plt.show()
        print(table.round(decimals=3).to_latex(float_format="%.4f"))

    elif run == 'pricing_exercise':

        S0 = 100

        fig = plt.figure(figsize=(11,8))

        for scheme in ['Euler-Maruyama', 'Milstein', 'Runge-Kutta', 'Closed-Form']:

            print(scheme)

            params = {
                "S0": S0,
                "T": 1,
                "sigma": 0.2,
                "rf": 0.05,
                "n_steps": 1200,
                "n_sims": 10000,
                "seed": 2023,
                "scheme": scheme
            }

            if scheme == 'Closed-form':
                params['scheme'] = None

            model = GeometricBrownianMotionSimulator(**params)
            model.simulate()
            model.get_theoretical_moments()

            calls = []
            puts = []
            strikes = []

            for k in np.arange(0.05, 2.10, 0.05):
                K = k * 100

                option_params = {
                    'strike': 'fixed',
                    'S0': S0,
                    'averaging': 'discrete',
                    'average': 'arithmetic',
                    'reset': 12,
                    'K': K
                }

                pricer = AsianOptionPricer(**option_params)
                call_price, put_price = pricer.get_value_mc(S=model.S, r=0.05, T=1)
                print(f'Asian call option: {call_price:0.4f}')
                print(f'Asian put option: {put_price:0.4f}')
                calls.append(call_price)
                puts.append(put_price)
                strikes.append(K)

            plt.plot(strikes, calls, label=f"{scheme}, Asian {option_params['average']} calls, {option_params['averaging']}", ls=':', marker='o')
            plt.plot(strikes, puts, label=f"{scheme}, Asian {option_params['average']} puts, {option_params['averaging']}", ls=':', marker='o')

        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':

    main(run='simulation_exercise')


