import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'Glass.csv'
data = pd.read_csv(file_path)

#Task 1 Проаналізуйте та підготуйте набір даних до класифікації
print()
print("Task 1")
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
sns.pairplot(data, hue="Type")
plt.show()




#Task 3 Навчіть класифікатор на тренувальному наборі даних. Застосуйте класифікатор до тестового набору даних.
print()
print("Task 3")
X = data.drop('Type', axis=1)
y = data['Type']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Точність класифікатору: {accuracy:.2f}')




#Task 4 Оцініть класифікатор
print()
print("Task 4")
print(classification_report(y_test, predictions))




#Task 5 Побудуйте матрицю неточностей.
confusion_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 7), yticklabels=range(1, 7))
plt.xlabel('Предсказані класи')
plt.ylabel('Справжні класи')
plt.title('Матриця неточностей')
plt.show()




#Task 6 Виконайте масштабування (нормалізацію) функції та проаналізуйте, чи вплинуло це на продуктивність класифікатора.
print()
print("Task 6")
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

knn_classifier_standardized = KNeighborsClassifier(n_neighbors=3)
knn_classifier_standardized.fit(X_train_standardized, y_train)

y_pred_knn_standardized = knn_classifier_standardized.predict(X_test_standardized)

accuracy_standardized = accuracy_score(y_test, y_pred_knn_standardized)
print(f"Точність класифікатора після нормалізації: {accuracy_standardized}")




#Task 7 Підберіть параметри класифікатора, за якого найбільша часткаправильних передбачень серед тестових даних.
print()
print("Task 7")

param_grid = {
 'n_neighbors': [2, 4, 8, 16, 32],
 'weights': ['uniform', 'distance'],
 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'leaf_size': [5, 10, 15, 20, 25, 30, 35, 40]
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_standardized, y_train)

print(f"Найкращі параметри: {grid_search.best_params_}")
print(f"Найкраща точність: {grid_search.best_score_}")




#Task 9 Напишіть власну функцію, яка розраховує евклідову відстань між двома об’єктами.
print()
print("Task 9")

def euclidean_distance(instance1, instance2):
    return np.linalg.norm(instance1 - instance2)

class_1 = X_test[y_test == 1].head(3)
class_2 = X_test[y_test == 2].head(3)
class_3 = X_test[y_test == 3].head(3)
class_5 = X_test[y_test == 5].head(3)

nearest_neighbors_class_1 = [X_train.iloc[np.argmin([euclidean_distance(class_1.iloc[i], x) for _, x in X_train.iterrows()])] for i in range(3)]
nearest_neighbors_class_2 = [X_train.iloc[np.argmin([euclidean_distance(class_2.iloc[i], x) for _, x in X_train.iterrows()])] for i in range(3)]
nearest_neighbors_class_3 = [X_train.iloc[np.argmin([euclidean_distance(class_3.iloc[i], x) for _, x in X_train.iterrows()])] for i in range(3)]
nearest_neighbors_class_5 = [X_train.iloc[np.argmin([euclidean_distance(class_5.iloc[i], x) for _, x in X_train.iterrows()])] for i in range(3)]

def print_results(class_name, class_data, nearest_neighbors):
    print(f"{class_name}:")
    for i, (obj, neighbor) in enumerate(zip(class_data.values, nearest_neighbors)):
        print(f" Об'єкт {i + 1}:")
        print(f"  RI: {obj[0]}, Na: {obj[1]}, Mg: {obj[2]}, Al: {obj[3]}, Si: {obj[4]}, K: {obj[5]}, Ca: {obj[6]}, Ba: {obj[7]}, Fe: {obj[8]}")
        print(f"  Найближчий сусід:")
        print(f"  RI: {neighbor[0]}, Na: {neighbor[1]}, Mg: {neighbor[2]}, Al: {neighbor[3]}, Si: {neighbor[4]}, K: {neighbor[5]}, Ca: {neighbor[6]}, Ba: {neighbor[7]}, Fe: {neighbor[8]}")

print_results("Клас 1", class_1, nearest_neighbors_class_1)
print_results("Клас 2", class_2, nearest_neighbors_class_2)
print_results("Клас 3", class_3, nearest_neighbors_class_3)
print_results("Клас 5", class_5, nearest_neighbors_class_5)




