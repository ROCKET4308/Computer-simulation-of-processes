import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'Glass.csv'
data = pd.read_csv(file_path)

#Task 1
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

#Task 2 матриця графіків розсіювання
sns.pairplot(data, hue="Type")  # Побудова матриці графіків розсіювання. Опція hue="Type" дозволяє розфарбувати графіки розсіювання відповідно до класів, що допомагає візуалізувати розподіл класів.

plt.show()



#Task 3
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
print(f'Точність класифікатору: {accuracy:.2f}')



#Task 4
# Виведення звіту з класифікації
print(classification_report(y_test, predictions))



#Task 5
# Побудова матриці неточностей
confusion_mat = confusion_matrix(y_test, predictions)
# Візуалізація матриці неточностей
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 7), yticklabels=range(1, 7))
plt.xlabel('Предсказані класи')
plt.ylabel('Справжні класи')
plt.title('Матриця неточностей')
plt.show()


#Task 6
#Стандартизація даних
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)
# Ініціалізація та навчання класифікатора k-найближчих сусідів на стандартизованих даних
knn_classifier_standardized = KNeighborsClassifier(n_neighbors=3)
knn_classifier_standardized.fit(X_train_standardized, y_train)
# Прогнозування класів для стандартизованого тестового набору
y_pred_knn_standardized = knn_classifier_standardized.predict(X_test_standardized)
# Оцінка точності класифікатора k-найближчих сусідів на стандартизованих даних
accuracy_standardized = accuracy_score(y_test, y_pred_knn_standardized)
print(f"Точність класифікатора після нормалізації: {accuracy_standardized}")


#Task 7
# Підготовка параметрів для перебору
param_grid = {
 'n_neighbors': [3, 5, 7, 9, 11],
 'weights': ['uniform', 'distance'],
 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'leaf_size': [10, 20, 30, 40]
}
# Ініціалізація класифікатора k-найближчих сусідів
knn = KNeighborsClassifier()
# Пошук оптимальних параметрів
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1) # cv вказує кількість згорток у перехресній перевірці
# Навчання класифікатора з використанням перехресної перевірки та пошук оптимальних параметрів
grid_search.fit(X_train_standardized , y_train)
# Виведення найкращих параметрів та їх впливу на точність
print(f"Найкращі параметри: {grid_search.best_params_}")
print(f"Найкраща точність: {grid_search.best_score_}")