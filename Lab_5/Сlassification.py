import pandas as pd


file_path = 'Glass.csv'
data = pd.read_csv(file_path)


print("Статистична інформація про числові дані з DataFrame:")
print(data.describe())


total_records = len(data)
print(f"Загальна кількість об'єктів у вибірці: {total_records}")


features_count = data.shape[1] - 1
features = data.columns[:-1]
print(f"Кількість ознак для класифікації: {features_count}")
print(f"Назви ознак: {', '.join(features)}")


missing_values = data.isnull().sum().sum()
if missing_values == 0:
    print("Пропущених значень немає.")
else:
    print(f"Кількість пропущених значень: {missing_values}")


duplicates = data.duplicated().sum()
if duplicates == 0:
    print("Дублікатів записів немає.")
else:
    print(f"Кількість дублікатів записів: {duplicates}")


instances_per_class = data['Type'].value_counts()
print("Екземпляри у кожному класі:")
print(instances_per_class)
