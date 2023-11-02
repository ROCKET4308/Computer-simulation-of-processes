import numpy as np
import pandas as pd
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

    def display_orders(self, queue, start_time_minutes):
        hours = start_time_minutes // 60
        minutes = start_time_minutes % 60
        print("-" * 50)
        print(f"Замовлення на піцу в {hours:02}:{minutes:02}:")
        for index, row in queue.iterrows():
            print()
            hours_order = row['Order Time'] // 60
            minutes_order = row['Order Time'] % 60
            pizza_time_minutes = row['Pizza Time']
            print(f"Час замовлення:  {hours_order:02}:{minutes_order:02}")
            print(f"Тип замовлення: {row['Pizza Type']}")
            print(f"Час виготовлення: {pizza_time_minutes:02} хвилин")
            print(f"Вартість: {row['Pizza Price']} умовних одиниць")
        print("-" * 50)

    def simulate(self):
        simulation_duration_minutes = 1320
        time_interval_minutes = 5
        np.random.seed(0)
        rs = np.random.RandomState(0)

        order_statistics = [0] * 24
        hourly_profit = [0] * 24
        busy_ovens = 0

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

                    if len(self.queue) < self.max_queue_length:
                        self.queue.append({
                            "Order Time": start_time_minutes,
                            "Pizza Type": pizza_type,
                            "Pizza Time": pizza_time,
                            "Pizza Price": pizza_price,
                        })
                        busy_ovens += 1
                        hourly_revenue += pizza_price

            orders_to_remove = []
            for order in self.queue:
                order_start_time = order["Order Time"]
                pizza_time = order["Pizza Time"]
                if start_time_minutes >= order_start_time + pizza_time:
                    orders_to_remove.append(order)

            for order in orders_to_remove:
                self.queue.remove(order)
                busy_ovens -= 1

            current_hour = start_time_minutes // 60
            order_statistics[current_hour] += num_orders
            hourly_profit[current_hour] += hourly_revenue

            orders_df = pd.DataFrame(self.queue)
            self.display_orders(orders_df, start_time_minutes)

            start_time_minutes += time_interval_minutes

        start_hour = 9
        end_hour = 22
        hours = range(start_hour, end_hour)  # Оновлені години

        # Графік кількості замовлень
        plt.figure(figsize=(12, 6))
        plt.bar(hours, order_statistics[start_hour:end_hour], tick_label=hours, color='skyblue')
        plt.xlabel('Година дня')
        plt.ylabel('Кількість замовлень')
        plt.title('Частота замовлень піц протягом дня')
        plt.xticks(hours)
        plt.grid(axis='y')
        plt.show()

        # Графік прибутку в кожну годину
        plt.figure(figsize=(10, 6))
        plt.bar(hours, hourly_profit[start_hour:end_hour], color='yellow')
        plt.title('Прибуток піцерії в кожну годину')
        plt.xlabel('Година')
        plt.ylabel('Прибуток')
        plt.xticks(hours)
        plt.grid(axis='y')
        plt.show()


pizzeria = Pizzeria()
pizzeria.simulate()

