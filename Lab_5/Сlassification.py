import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

sns.pairplot(data, hue="Type")  # Побудова матриці графіків розсіювання. Опція hue="Type" дозволяє розфарбувати графіки розсіювання відповідно до класів, що допомагає візуалізувати розподіл класів.

plt.show()

# Розділення на вхідні та вихідні дані
X = data.drop('Type', axis=1)  # Ознаки
y = data['Type']  # Цільова змінна

# Перетворення цільової змінної у числові мітки
le = LabelEncoder()
y = le.fit_transform(y)

# Розділення на тренувальний та тестовий набори даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація та навчання класифікатора
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Прогнозування на тестових даних
predictions = classifier.predict(X_test)

# Оцінка точності моделі
accuracy = accuracy_score(y_test, predictions)
print(f'Точність: {accuracy:.2f}')

# Виведення звіту з класифікації
print(classification_report(y_test, predictions))

