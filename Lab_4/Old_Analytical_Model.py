import math
import numpy as np
from prettytable import PrettyTable


class Pizzeria:
    def __init__(self):
        self.max_queue_length = 10
        self.num_ovens = 10
        self.oven_cost_per_hour = 20
        self.standard_pizza_time = 10
        self.custom_pizza_time = 15
        self.standard_pizza_price = np.random.randint(10, 15)
        self.custom_pizza_price = np.random.randint(15, 30)
        self.queue = []
        self.ovens_list = []
        self.intervals = [(540, 660), (660, 900), (900, 1200), (1200, 1320)]
        self.probabilities = [0.3, 0.5, 0.9, 0.7]
        self.order_statistics = [0] * 24
        self.hourly_profit = [0] * 24

    def simulate(self):
        simulation_duration_minutes = 1320
        time_interval_minutes = 5
        np.random.seed(0)
        rs = np.random.RandomState(0)

        start_time_minutes = 540

        while start_time_minutes < simulation_duration_minutes:
            interval = None
            for i, (start, end) in enumerate(self.intervals):
                if start <= start_time_minutes < end:
                    interval = self.intervals[i]
                    break

            hourly_revenue = 0

            if interval:
                num_orders = rs.binomial(1, self.probabilities[i])
                for _ in range(num_orders):
                    if rs.random() < 0.7:
                        pizza_type = "Standard"
                        pizza_time = self.standard_pizza_time
                        pizza_price = self.standard_pizza_price
                    else:
                        pizza_type = "Custom"
                        pizza_time = self.custom_pizza_time
                        pizza_price = self.custom_pizza_price

                    if len(self.ovens_list) < self.num_ovens:
                        self.ovens_list.append({
                            "Order Time": start_time_minutes,
                            "Pizza Type": pizza_type,
                            "Pizza Time": pizza_time,
                            "Pizza Price": pizza_price,
                        })
                        hourly_revenue += pizza_price

                    elif len(self.queue) < self.max_queue_length:
                        self.queue.append({
                            "Order Time": start_time_minutes,
                            "Pizza Type": pizza_type,
                            "Pizza Time": pizza_time,
                            "Pizza Price": pizza_price,
                        })

            for oven in self.ovens_list:
                if start_time_minutes >= oven["Order Time"] + oven["Pizza Time"]:
                    self.ovens_list.remove(oven)

            for order in self.queue:
                if len(self.ovens_list) < self.num_ovens:
                    self.ovens_list.append(order)
                    hourly_revenue += order["Pizza Price"]
                self.queue.remove(order)

            current_hour = start_time_minutes // 60
            self.order_statistics[current_hour] += num_orders
            self.hourly_profit[current_hour] += hourly_revenue

            start_time_minutes += time_interval_minutes

    def analytical(self):
        time_intervals = range(9, 22)
        oven_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for time_interval in time_intervals:
            table = PrettyTable()
            table.field_names = ["Кількість включених печей", *[f"c = {c}" for c in oven_counts]]

            for characteristic in ["Інтенсивність вхідного потоку (lambda)", "Середній час обслуговування одного клієнта (t_obslug)",
                                    "Інтенсивність обслуговування (mu)", "Завантаженість системи (ρ)",
                                   "Граничні ймовірності (q0)", "Ймовірність того, що вимога потрапить у чергу (pq)", "Ймовірність відмови (p_vidmovy)",
                                    "Відносна пропускна здатність (Q)", "Абсолютна пропускна здатність (A)",
                                    "Середня кількість зайнятих каналів (k̅ зайнятих)",
                                   "Середня кількість вимог в черзі (Lq)", "Середня кількість вимог в системі (Ls)",
                                   "Середній час перебування вимог у черзі (Wq)",
                                   "Середній час перебування вимог у системі (Ws)", "Виручка (D)"]:
                row_data = [characteristic]
                for oven_count in oven_counts:
                    q0, p_vidmovy, pq, Q, A, k_occupied, Lq, Ls, Wq, Ws, D, p, lambda_value, mu, t_service = self.calculate_system_characteristics_for_period(time_interval, oven_count)

                    if characteristic == "Інтенсивність вхідного потоку (lambda)":
                        row_data.append(f"{lambda_value:.10f}")
                    elif characteristic == "Середній час обслуговування одного клієнта (t_obslug)":
                        row_data.append(f"{t_service:.10f}")
                    elif characteristic == "Інтенсивність обслуговування (mu)":
                        row_data.append(f"{mu:.10f}")
                    elif characteristic == "Завантаженість системи (ρ)":
                        row_data.append(f"{p:.10f}")
                    elif characteristic == "Граничні ймовірності (q0)":
                        row_data.append(f"{q0:.10f}")
                    elif characteristic == "Ймовірність того, що вимога потрапить у чергу (pq)":
                        row_data.append(f"{pq:.10f}")
                    elif characteristic == "Ймовірність відмови (p_vidmovy)":
                        row_data.append(f"{p_vidmovy:.10f}")
                    elif characteristic == "Відносна пропускна здатність (Q)":
                        row_data.append(f"{Q:.10f}")
                    elif characteristic == "Абсолютна пропускна здатність (A)":
                        row_data.append(f"{A:.10f}")
                    elif characteristic == "Середня кількість зайнятих каналів (k̅ зайнятих)":
                        row_data.append(f"{k_occupied:.10f}")
                    elif characteristic == "Середня кількість вимог в черзі (Lq)":
                        row_data.append(f"{Lq:.10f}")
                    elif characteristic == "Середня кількість вимог в системі (Ls)":
                        row_data.append(f"{Ls:.10f}")
                    elif characteristic == "Середній час перебування вимог у черзі (Wq)":
                        row_data.append(f"{Wq:.10f}")
                    elif characteristic == "Середній час перебування вимог у системі (Ws)":
                        row_data.append(f"{Ws:.10f}")
                    elif characteristic == "Виручка (D)":
                        row_data.append(f"{D:.10f}")

                table.add_row(row_data)

            print(f"Результати аналізу в {time_interval}:00")
            print(table)
            print("\n")

    def calculate_system_characteristics_for_period(self, time_interval, oven_count):

        lambda_value = self.order_statistics[time_interval]

        t_service = 60 / self.order_statistics[time_interval]
        t_service = t_service / 60

        mu = 1 / t_service

        p = lambda_value / mu

        c = oven_count

        m = lambda_value - c

        q0 = 1.0
        for i in range(c):
            q0 += (p ** i) / math.factorial(i)

        q0 += (m * (p ** (c + 1))) / (c * math.factorial(c))
        q0 = 1.0 / q0

        probabilities = [0] * (c + 1)
        for i in range(c):
            probabilities[i] = ((p ** i) / math.factorial(i)) * q0
        probabilities[c] = (p ** c) / math.factorial(c) * q0

        if p != c:
            pq = (((p ** (c)) * (1 - ((p ** (m)) / (c ** (m))))) / (math.factorial(c) * (1 - (p / c)))) * q0
        else:
            pq = ((p ** (c))/math.factorial(c)) * m * q0

        p_vidmovy = ((p ** (c + m)) / ((c ** (m)) * math.factorial(c))) * q0

        Q = 1 - p_vidmovy

        A = lambda_value * Q

        k_occupied = p * Q

        Lq = (((p ** (c + 1)) * m * (m + 1)) / (math.factorial(c) * c * 2)) * q0
        Ls = Lq + k_occupied
        Wq = Lq / lambda_value
        Ws = Ls / lambda_value

        profit = self.hourly_profit[time_interval]

        D = profit - 20 * c


        return q0, p_vidmovy, pq, Q, A, k_occupied, Lq, Ls, Wq, Ws, D, p, lambda_value, mu, t_service


pizzeria = Pizzeria()
pizzeria.simulate()
pizzeria.analytical()