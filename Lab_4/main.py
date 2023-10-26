import numpy as np
import pandas as pd


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

    def display_orders(self, queue):
        print("Замовлення на піцу:")
        print("-" * 50)
        for index, row in queue.iterrows():
            hours = row['Order Time'] // 60
            minutes = row['Order Time'] % 60
            pizza_time_minutes = row['Pizza Time']
            print(f"Час:  {hours:02}:{minutes:02}")
            print(f"Тип замовлення: {row['Pizza Type']}")
            print(f"Час виготовлення: {pizza_time_minutes:02}")
            print(f"Вартість: {row['Pizza Price']} умовних одиниць")
            print()

    def simulate(self):
        simulation_duration_minutes = 1320
        time_interval_minutes = 1
        np.random.seed(0)
        rs = np.random.RandomState(0)

        intervals = [(540, 660), (660, 900), (900, 1200), (1200, 1320)]
        probabilities = [0.3, 0.5, 0.9, 0.7]

        start_time_minutes = 540

        while start_time_minutes < simulation_duration_minutes:
            interval = None
            for i, (start, end) in enumerate(intervals):
                if start <= start_time_minutes < end:
                    interval = intervals[i]
                    break

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

                    if len(self.queue) < self.max_queue_length:
                        self.queue.append({
                            "Order Time": start_time_minutes,
                            "Pizza Type": pizza_type,
                            "Pizza Time": pizza_time,
                            "Pizza Price": pizza_price,
                        })

            orders_to_remove = []
            for order in self.queue:
                order_start_time = order["Order Time"]
                pizza_time = order["Pizza Time"]
                if start_time_minutes >= order_start_time + pizza_time:
                    orders_to_remove.append(order)

            for order in orders_to_remove:
                self.queue.remove(order)

            orders_df = pd.DataFrame(self.queue)
            self.display_orders(orders_df)

            start_time_minutes += time_interval_minutes




pizzeria = Pizzeria()
pizzeria.simulate()