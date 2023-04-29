import random
import math

def f(x):
    return math.exp(-x**2)

# define the limits of integration
a = 0
b = float('inf')

# set the number of points to use in the simulation
n = 100000

# initialize the sum of function values and the count of points inside the region
sum = 0
count = 0

# generate n random points and use them to estimate the integral
for i in range(n):
    # generate a random value of x between a and b
    x = random.uniform(a, b)
    # generate a random value of y between 0 and the maximum value of f(x)
    y = random.uniform(0, 1/math.sqrt(math.pi))
    # check if the point (x, y) is inside the region of integration
    if y <= f(x):
        # if the point is inside, add f(x) to the sum and increment the count
        sum += f(x)
        count += 1

# check if count is zero
if count == 0:
    integral = 0
else:
    # estimate the integral as the average of the function values times the area of the region
    integral = sum/count * (b-a) * 1/math.sqrt(math.pi)

print("Monte Carlo approximation of the integral: ", integral)