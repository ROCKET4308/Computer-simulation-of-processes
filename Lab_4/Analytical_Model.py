import numpy as np
import matplotlib.pyplot as plt


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
        self.ovens_list = []

    def simulate(self):
        simulation_duration_minutes = 1325
        time_interval_minutes = 5
        np.random.seed(0)
        rs = np.random.RandomState(0)

        order_statistics = [0] * 24
        hourly_profit = [0] * 24


        intervals = [(540, 660), (660, 900), (900, 1200), (1200, 1320)]
        probabilities = [0.3, 0.5, 0.9, 0.7]

        start_time_minutes = 540

        while start_time_minutes < simulation_duration_minutes:
            interval = None
            for i, (start, end) in enumerate(intervals):
                if start <= start_time_minutes < end:
                    interval = intervals[i]
                    break

            hourly_revenue = 0

            if interval:
                num_orders = rs.binomial(1, probabilities[i])
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
            order_statistics[current_hour] += num_orders
            hourly_profit[current_hour] += hourly_revenue

            start_time_minutes += time_interval_minutes



Pizzeria().simulate()
