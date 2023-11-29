from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import numpy as np

def scalar_function(x):
    return (x[0] - 6) ** 2 + (x[1] - 4) ** 2 + x[0] * x[1]

result = minimize(scalar_function, x0=[0, 0])
min_x, min_y = result.x
min_z = result.fun
print(f"Мінімум функції: x = {min_x}, y = {min_y}, f(x, y) = {min_z}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
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


plt.figure(figsize=(16, 5))
plt.subplot(131)
contour_yoz = plt.contour(Y, Z, X, levels=20, cmap='viridis')
plt.scatter(min_y, min_z, color='red', marker='o', label='Minimum')
plt.title('Contour Plot YOZ')
plt.xlabel('y')
plt.ylabel('f(x, y)')
plt.colorbar(contour_yoz)
plt.legend()

plt.subplot(132)
contour_xoy = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.scatter(min_x, min_y, color='red', marker='o', label='Minimum')
plt.title('Contour Plot XOY')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(contour_xoy)
plt.legend()

plt.subplot(133)
contour_xoz = plt.contour(X, Z, Y, levels=20, cmap='viridis')
plt.scatter(min_x, min_z, color='red', marker='o', label='Minimum')
plt.title('Contour Plot XOZ')
plt.xlabel('x')
plt.ylabel('f(x, y)')
plt.colorbar(contour_xoz)
plt.legend()
plt.show()


