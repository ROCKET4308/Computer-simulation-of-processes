import numpy as np


class Pizzeria:
    def __init__(self):
        self.num_ovens = 10
        self.max_queue_length = 10
        self.oven_cost_per_hour = 20
        self.standard_pizza_time = 10
        self.custom_pizza_time = 15
        self.standard_pizza_price = np.random.randint(10, 15)
        self.custom_pizza_price = np.random.randint(15, 30)
        self.queue = []
        self.oven_timers = [0] * 10

    def add_order(self, order_type):
        if len(self.queue) < self.max_queue_length:
            self.queue.append(order_type)

    def simulate(self, time_interval_minutes, simulation_duration_minutes):
        np.random.seed(0)
        rs = np.random.RandomState(0)

        intervals = [(540, 660), (660, 900), (900, 1200), (1200, 1320)]
        probabilities = [0.3, 0.5, 0.9, 0.7]

        start_time_minutes = 540
        orders = []

        while start_time_minutes < simulation_duration_minutes:
            interval = None
            for i, (start, end) in enumerate(intervals):
                if start <= start_time_minutes < end:
                    interval = intervals[i]
                    break

            if interval:
                num_orders = rs.binomial(1, probabilities[i])
                for _ in range(num_orders):
                    order_type = "Standard" if rs.random() < 0.7 else "Custom"
                    self.add_order(order_type)

            # Вивести чергу замовлень
            print(f"Current Queue: {self.queue}")

            for i in range(self.num_ovens):
                if self.oven_timers[i] > 0:
                    self.oven_timers[i] -= time_interval_minutes
                    if self.oven_timers[i] == 0:
                        if self.queue:
                            orders.append(self.queue.pop(0))

            for i in range(self.num_ovens):
                if self.oven_timers[i] == 0 and self.queue:
                    if self.queue[0] == "Standard":
                        self.oven_timers[i] = self.standard_pizza_time
                    else:
                        self.oven_timers[i] = self.custom_pizza_time

            start_time_minutes += time_interval_minutes

        return orders


pizzeria = Pizzeria()
simulation_duration_minutes = 1320  # 1 day
time_interval_minutes = 5

orders = pizzeria.simulate(time_interval_minutes, simulation_duration_minutes)