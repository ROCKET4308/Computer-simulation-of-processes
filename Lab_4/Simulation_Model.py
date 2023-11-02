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

    def display_status(self, ovens_list, queue, start_time_minutes):
        hours = start_time_minutes // 60
        minutes = start_time_minutes % 60
        print("-" * 50)
        print(f"Черга на піцу в {hours:02}:{minutes:02}:")
        for que in queue:
            print()
            hours_order = que['Order Time'] // 60
            minutes_order = que['Order Time'] % 60
            pizza_time_minutes = que['Pizza Time']
            print(f"Час замовлення:  {hours_order:02}:{minutes_order:02}")
            print(f"Тип замовлення: {que['Pizza Type']}")
            print(f"Час виготовлення: {pizza_time_minutes:02} хвилин")
            print(f"Вартість: {que['Pizza Price']} умовних одиниць")

        print()
        print(f"Піци які знаходяться в печях:")
        for oven in ovens_list:
            print()
            hours_order = oven['Order Time'] // 60
            minutes_order = oven['Order Time'] % 60
            pizza_time_minutes = oven['Pizza Time']
            print(f"Час замовлення:  {hours_order:02}:{minutes_order:02}")
            print(f"Тип замовлення: {oven['Pizza Type']}")
            print(f"Час виготовлення: {pizza_time_minutes:02} хвилин")
            print(f"Вартість: {oven['Pizza Price']} умовних одиниць")

        busy_oven_count = len(
            [oven for oven in ovens_list if start_time_minutes <= oven["Order Time"] + oven["Pizza Time"]])
        free_oven_count = self.num_ovens - busy_oven_count

        print()
        print(f"Стан печей:")
        print(f"Зайнято печей: {busy_oven_count}")
        print(f"Вільно печей: {free_oven_count}")
        print("-" * 50)

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

            self.display_status(self.ovens_list, self.queue, start_time_minutes)
            start_time_minutes += time_interval_minutes


        start_hour = 9
        end_hour = 22
        hours = range(start_hour, end_hour)

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

        profit = 0
        for prof in hourly_profit:
            profit += prof
        total_profit = profit - 10 * 20 * (22 - 9)
        print(f"Profit without oven payments: {profit}")
        print(f"Total profit: {total_profit}")



Pizzeria().simulate()