from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import numpy as np

def scalar_function(x):
    return x**2 + 10*x + 9

result = minimize(scalar_function, x0=0)
min_x = result.x[0]
min_y = result.fun

print(f"Мінімум функції: x = {min_x}, f(x) = {min_y}")

x_vals = np.linspace(-20, 20, 1000)
y_vals = scalar_function(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Objective Function')
plt.scatter(min_x, min_y, color='red', marker='o', label='Minimum')
plt.title('One-dimensional Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

result_global = differential_evolution(scalar_function, bounds=[(-10, 10)])
min_x_global = result_global.x
min_y_global = result_global.fun

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Objective Function')
plt.scatter(min_x, min_y, color='red', marker='o', label='Local Minimum')
plt.scatter(min_x_global, min_y_global, color='blue', marker='x', label='Global Minimum')
plt.title('One-dimensional Function with Local and Global Minima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

