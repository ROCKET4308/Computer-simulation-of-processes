from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np

def scalar_function(x):
    return x**2 + 5*x + 7

minima = minimize_scalar(scalar_function, bracket=[-10, 10]).x

print("Мінімум знаходиться при x =", minima)
print("Значення функції у мінімумі:", scalar_function(minima))

x_vals = np.linspace(-10, 10, 1000)
y_vals = scalar_function(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Функція')
plt.scatter(minima, scalar_function(minima), color='red', label='Мінімум')
plt.xlabel('x')
plt.ylabel('Значення функції')
plt.title('Графік функції з відміченим мінімумом')
plt.legend()
plt.grid(True)
plt.show()



