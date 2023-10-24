import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.ticker import FuncFormatter

#графік зміни швидкості при зміні маси
def massChangeGraphik():
    G = 6.67430e-11
    mass_of_earth = 5.972e24
    initial_velocity = np.sqrt(G * mass_of_earth / 6371000)
    planet_masses = np.linspace(0.01 * mass_of_earth, 100.0 * mass_of_earth, 1000)
    escape_velocities = np.sqrt(G * planet_masses / 6371000)
    plt.figure(figsize=(10, 6))
    plt.plot(planet_masses, escape_velocities, label="Необхідна швидкість")
    plt.axhline(initial_velocity, color='red', linestyle='--', label="Перша космічна швидкість для стандартних умов на Земля")
    plt.xlabel("Маса планети (кг)")
    plt.ylabel("Перша космічна швидкість (м/с)")
    plt.title("Зміна необхідної швидкості для першої космічної швидкості залежно від маси планети")
    plt.legend()
    plt.grid(True)
    #def format_func(value, tick_number):
     #   return f'{value / 1e18:.0f} қн. кг'
    #plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.show()

#графік зміни швидкості при зміні радіуса
def radiusChangeGraphik():
    G = 6.67430e-11  # Гравітаційна постійна, м^3/(кг*с^2)
    M_earth = 5.972e24  # Маса Землі, кг
    def escape_velocity(radius):
        return np.sqrt(G * M_earth / radius)
    initial_velocity = np.sqrt(G * M_earth / 6371000)
    radius_values = np.arange(2e6, 7e7, 1e4) # Починаємо з 1 км і закінчуємо 10 тис. км
    velocity_values = escape_velocity(radius_values)
    plt.figure(figsize=(15, 8))
    plt.plot(radius_values/ 1000, velocity_values)
    plt.axhline(initial_velocity, color='red', linestyle='--',label="Перша космічна швидкість для стандартних умов на Земля")
    plt.title('Зміна необхідної швидкості для першої космічної швидкості залежно від радіусу планети')
    plt.xlabel('Радіус планети (км)')
    plt.ylabel('Перша космічна швидкість (м/с)')
    plt.grid(True)
    plt.legend()
    def format_func(value, tick_number):
        if value >= 1e6:
            return f'{value * 1e-6:.0f} млн км'
        elif value >= 1e3:
            return f'{value * 1e-3:.0f} тис. км'
        else:
            return f'{value:.0f} км'
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.show()

#таблиця назва планети,  маса, рідіус, швидкість
def speedTable():
    G = 6.67430e-11
    masses = {
        "Меркурій": 3.3011e23,
        "Венера": 4.8675e24,
        "Земля": 5.9723e24,
        "Марс": 6.4171e23,
        "Юпітер": 1.8982e27,
        "Сатурн": 5.6834e26,
        "Уран": 8.6810e25,
        "Нептун": 1.02413e26
    }
    radii = {
        "Меркурій": 2.4397e6,
        "Венера": 6.0518e6,
        "Земля": 6.371e6,
        "Марс": 3.3895e6,
        "Юпітер": 6.9911e7,
        "Сатурн": 5.8232e7,
        "Уран": 2.5362e7,
        "Нептун": 2.4622e7
    }
    def calculate_escape_velocity(planet_name):
        mass = masses[planet_name]
        radius = radii[planet_name]
        escape_velocity = np.sqrt(G * mass / radius)
        return escape_velocity
    table = PrettyTable()
    table.field_names = ["Планета", "Маса (кг)", "Радіус (м)", "Перша космічна швидкість (м/с)"]
    for planet_name in masses.keys():
        mass = masses[planet_name]
        radius = radii[planet_name]
        escape_velocity = calculate_escape_velocity(planet_name)
        table.add_row([planet_name, f"{mass:.2e}", f"{radius:.2e}", f"{escape_velocity:.2f}"])
    print(table)

#діаграми який відсоток впливу маси а який радіусу
def diagramOfSpeedForMass():
    masses = np.array([3.3011e23, 4.8675e24, 5.972e24,  6.39e23, 1.898e27, 5.683e26, 8.681e25])
    radius_earth = 6.371e6
    G = 6.67430e-11
    escape_velocities = np.sqrt(G * masses / radius_earth)
    planet_names = ['Меркурій', 'Венера', 'Земля',  'Марс', 'Юпітер', 'Сатурн', 'Уран']
    plt.figure(figsize=(10, 6))
    plt.barh(planet_names, escape_velocities, color='skyblue')
    plt.xlabel('Перша космічна швидкість (м/с)')
    plt.title('Перша космічна швидкість для планет сонячної системи з різною масою але з однаковим радіусом')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.gca().invert_yaxis()
    plt.show()


def diagramOfSpeedForRadius():
    G = 6.67430e-11
    mass_of_earth = 5.972e24
    planet_radii = {
        'Меркурій': 2439700,
        'Венера': 6051800,
        'Земля': 6371000,
        'Марс': 3389500,
        'Юпітер': 69911000,
        'Сатурн': 58232000,
        'Уран': 25362000,
        'Нептун': 24622000
    }
    planet_speeds = {}
    for planet, radius in planet_radii.items():
        v = np.sqrt(G * mass_of_earth / radius)
        planet_speeds[planet] = v
    plt.figure(figsize=(10, 6))
    plt.bar(planet_speeds.keys(), planet_speeds.values())
    plt.xlabel('Планета')
    plt.ylabel('Перша космічна швидкість (м/с)')
    plt.title('Перша космічна швидкість для планет сонячної системи з різним радіусом але з однаковою масою')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    massChangeGraphik()
    radiusChangeGraphik()
    diagramOfSpeedForMass()
    diagramOfSpeedForRadius()
    speedTable()


