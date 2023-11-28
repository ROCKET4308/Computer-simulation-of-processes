from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

def scalar_function(x):
    return (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + x[0] * x[1]

bounds = [(-10, 10), (-10, 10)]

initial_guess = [0, 0]

result = minimize(scalar_function, initial_guess, bounds=bounds, method='L-BFGS-B')

print("Мінімум:", result.x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = scalar_function([X, Y])
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.scatter(result.x[0], result.x[1], scalar_function(result.x), color='red', s=100, label='Мінімум')
ax.set_title('3D Surface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.legend()
plt.show()


plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.scatter(result.x[0], result.x[1], color='red', label='Мінімум')
plt.legend()
plt.show()
