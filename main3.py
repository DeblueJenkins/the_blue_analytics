import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2023)

def h1(x):
    return (2*x + 1)**2 * 2

def h2(x):
    return np.exp(-(x/(1-x)) ** 2) * (1 / (1-x)) ** 2

def h3(x):
    c = np.sqrt(np.pi/2)
    first_term = np.exp(-1/(2*np.tan(np.pi*x)**2))
    second_term = 1 / np.tan(np.pi*x) ** 4
    third_term = 1 / np.sin(np.pi*x) ** 2
    return c * first_term * second_term * third_term


d = [(h1, 9-1/3),
     (h2, np.sqrt(np.pi)/2),
     (h3, 3)]

# d = [(h1, 9-1/3)]
# d = [(h1, 1/3)]

n_integrals = len(d)
my_range = np.linspace(10, 300) ** 2
my_range = my_range.astype(int)

for i in range(n_integrals):

    func, h = d[i]
    results = []
    ci_lower = []
    ci_upper = []

    for n in my_range:

        x_sim = np.random.uniform(size=n, low=0, high=1)
        h_hat = func(x_sim)
        se = 1/(n**2) * np.sum((h-h_hat) ** 2)

        mean = np.mean(h_hat)

        results.append(mean)

        ci_lower.append(mean - 1.65 * se)
        ci_upper.append(mean + 1.65 * se)

    plt.plot(my_range, results, color='black', label='estimate')
    plt.plot(my_range, ci_lower, color='black', label='5%', ls=':')
    plt.plot(my_range, ci_upper, color='black', label='95%', ls=':')
    plt.axhline(y=h, color='red', label='truth')
    plt.grid(True)
    plt.savefig(fr'plots\monte_carlo_integration_{i}.png')
    plt.show()

