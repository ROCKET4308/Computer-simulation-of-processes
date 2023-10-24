import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Pizzeria:
    def __init__(self):
        self.standard_pizza_time = 10
        self.custom_pizza_time = 15
        self.standard_pizza_price = np.random.randint(10, 15)
        self.custom_pizza_price = np.random.randint(15, 30)

    def display_orders(self, orders_df):
        print("Замовлення на піцу:")
        print("-" * 50)
        for index, row in orders_df.iterrows():
            hours = row['Time'] // 60
            minutes = row['Time'] % 60

            order_time_hours = row['Order Time'] // 60
            order_time_minutes = row['Order Time'] % 60

            print(f"Час:  {hours:02}:{minutes:02}")
            print(f"Тип замовлення: {row['Order Type']}")
            print(f"Час виготовлення: {order_time_hours:02}:{order_time_minutes:02}")
            print(f"Вартість: {row['Order Price']} умовних одиниць")
            print("-" * 50)


def simulate_orders(pizzeria):
    time_interval_minutes = 5
    np.random.seed(0)
    rs = np.random.RandomState(0)

    intervals = [(540, 660), (660, 900), (900, 1200), (1200, 1320)]# TODO: переробити час
    probabilities = [0.3, 0.5, 0.9, 0.7]#TODO: переробити вірогідність

    start_time_minutes = 0
    orders = []

    while start_time_minutes < 1440:
        interval = None
        for i, (start, end) in enumerate(intervals):
            if start <= start_time_minutes < end:
                interval = intervals[i]
                break

        if interval:
            num_orders = rs.binomial(1, probabilities[i])
            for _ in range(num_orders):
                if rs.random() < 0.7:
                    order_type = "Standard"
                    order_time_minutes = pizzeria.standard_pizza_time
                    order_price = pizzeria.standard_pizza_price
                else:
                    order_type = "Custom"
                    order_time_minutes = pizzeria.custom_pizza_time
                    order_price = pizzeria.custom_pizza_price

                orders.append({
                    "Time": start_time_minutes,
                    "Order Type": order_type,
                    "Order Time": order_time_minutes,
                    "Order Price": order_price,
                })

        start_time_minutes += time_interval_minutes

    return pd.DataFrame(orders)

pizzeria = Pizzeria()
orders_df = simulate_orders(pizzeria)

pizzeria.display_orders(orders_df)

# TODO:
orders_per_hour = orders_df.groupby(orders_df['Time'] // 60).size()


plt.figure(figsize=(12, 6))
plt.bar(orders_per_hour.index, orders_per_hour.values, tick_label=orders_per_hour.index)
plt.xlabel('Година')
plt.ylabel('Кількість замовлень')
plt.title('Частота замовлень піц протягом дня')
plt.xticks(range(24))
plt.show()


hourly_profit = []
for hour in range(24):
    hour_orders = orders_df[(orders_df["Time"] >= hour * 60) & (orders_df["Time"] < (hour + 1) * 60)]
    profit = sum(hour_orders["Order Price"])
    hourly_profit.append(profit)



hours = range(24)
plt.figure(figsize=(12, 6))
plt.bar(hours, hourly_profit, tick_label=hours, color='orange')
plt.xlabel("Година")
plt.ylabel("Прибуток")
plt.title("Прибуток піцерії в кожну годину дня")
plt.show()


profits = []
time_intervals = list(range(0, 1440, 10))
def calculate_daily_profit(orders_df):
    return orders_df["Order Price"].sum()

for interval in time_intervals:
    profit = calculate_daily_profit(orders_df[orders_df["Time"] <= interval])
    profits.append(profit)


plt.plot(time_intervals, profits)
plt.xlabel("Час (хвилини)")
plt.ylabel("Загальний прибуток")
plt.title("Загальний прибуток за день у піцерії")
plt.grid(True)
plt.show()

