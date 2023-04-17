from simulator import GeometricBrownianMotionSimulator
from scipy.stats import describe
import numpy as np
from pricers import AsianOptionPricer
import matplotlib.pyplot as plt


def main(run='simulation_exercise'):
    print(run)

    if run == 'simulation_exercise':

        for scheme in [None, 'Euler-Maruyama', 'Milstein', 'Runge-Kutta']:

            print(f'scheme: {scheme}')

            params = {
                "S0": 100,
                "T": 1,
                "sigma": 0.2,
                "rf": 0.05,
                "n_steps": 5000,
                "n_sims": 100000,
                "seed": 2023,
                "scheme": scheme
            }


            model = GeometricBrownianMotionSimulator(**params)
            model.simulate()
            # model.plot_paths()
            # model.plot_dist()

            model.get_theoretical_moments()

            res = describe(model.S[:,-1])
            print(res)
            print(f"variance error: {np.round(np.abs(100 * (res.variance - model.theoretical_variance) / model.theoretical_variance), 4)}%")
            print(f"mean error: {np.round(np.abs(100 * (res.mean - model.theoretical_mean) / model.theoretical_mean), 4)}%")

        print(f'theoretical mean: {model.theoretical_mean}')
        print(f'theoretical variance: {model.theoretical_variance}')

    elif run == 'pricing_exercise':

        S0 = 100

        fig = plt.figure(figsize=(11,8))

        for scheme in ['Euler-Maruyama', 'Milstein', 'Runge-Kutta']:

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
                    'S': model.S,
                    'averaging': 'discrete',
                    'average': 'arithmetic'
                }

                pricer = AsianOptionPricer(**option_params)
                call_price, put_price = pricer.get_value(r=0.05, T=1, K=K)
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

    main('pricing_exercise')

