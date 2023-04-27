import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import colormaps


P = 3
Q = 2

f = lambda x: 4 * x ** 2

def alpha(h, P):
    return 1 + h/2 * P

def beta(h, Q):
    return -2 + h ** 2 * Q

def gamma(h, P):
    return 1 - h/2 * P

def y(x):
    term1 = 38.23189732 * np.exp(-x)
    term2 = -118.70318394 * np.exp(-2*x)
    polyterm = 2*x ** 2 - 6*x + 7
    return term1 + term2 + polyterm

# myA = np.array([[np.exp(-1), np.exp(-2)], [np.exp(-2), np.exp(-4)]])
# myb = np.array([-2, 3])
# myw = solve(myA, myb)

# cmap = colormaps['gray']
# num_shades = 10
# shades = np.linspace(-5, -2, num_shades)


fig = plt.figure(figsize=(11,8))
for cnt, n in enumerate(np.arange(10, 105, 5)):

    a = 1
    b = 2

    w0 = 1
    wn = 6

    h = (b - a) / n

    xi = np.array([a + i*h for i in range(n+1)])

    # Initialize matrices and vectors

    A_matrix = np.zeros((n,n))
    w_vector = np.zeros(n)
    b_vector = np.zeros(n)

    # Set first row

    A_matrix[0,0] = beta(h,Q)
    A_matrix[0,1] = alpha(h, P)

    b_vector[0] = -gamma(h,P) * w0 + h ** 2 * f(xi[0])

    # Set intermediate rows
    for i in np.arange(1, n-1):

        A_matrix[i, i-1] = gamma(h,P)
        A_matrix[i, i] = beta(h,Q)
        A_matrix[i, i+1] = alpha(h, P)

        b_vector[i] = h ** 2 * f(xi[i])

    # Set last row

    A_matrix[-1,-1] = beta(h, P)
    A_matrix[-1,-2] = gamma(h,P)

    b_vector[-1] = h ** 2 * f(xi[-1]) - alpha(h,P) * wn

    # Solve system

    w = solve(A_matrix, b_vector)



    if cnt == 0:
        plt.plot(xi, y(xi), label='analytic soln', color='red')
    plt.plot(xi, np.insert(w, 0, w0), label=f'n={n}', ls=':')

plt.legend()
plt.grid(True)
plt.show()