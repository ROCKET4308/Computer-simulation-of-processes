import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


file_path = 'Glass.csv'
data = pd.read_csv(file_path)
features = data.drop(columns=['Type'])

# Task 2
k_clusters = len(data['Type'].unique())
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
features['Cluster'] = kmeans.fit_predict(features)

cluster_stats = features.groupby('Cluster').describe().transpose()
print(cluster_stats)


# Task 3
# Вибірка параметрів для побудови графіків
selected_features = ["RI", "Na", "Mg", "Al"]

# Побудова графіків попарно
plt.figure(figsize=(15, 10))

for i in range(len(selected_features)):
    for j in range(len(selected_features)):
        plt.subplot(len(selected_features), len(selected_features), i * len(selected_features) + j + 1)
        sns.scatterplot(data=features, x=selected_features[i], y=selected_features[j], hue='Cluster', palette='viridis', s=50)
        plt.scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:, j], marker='x', s=200, linewidths=3, color='red')
        plt.title(f'{selected_features[i]} vs {selected_features[j]}')
        plt.xlabel(selected_features[i])
        plt.ylabel(selected_features[j])

plt.tight_layout()
plt.show()

# Task 4
instances_per_cluster = features['Cluster'].value_counts()
print("Instances per cluster:")
print(instances_per_cluster)

cluster_class_distribution = pd.crosstab(data['Type'], features['Cluster'])
print("\nClass distribution in each cluster:")
print(cluster_class_distribution)

# Task 5
WCSS = []
silhouette_scores = []
davies_bouldin_scores = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    WCSS.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, labels))
    davies_bouldin_scores.append(davies_bouldin_score(features, labels))


cluster_metrics = pd.DataFrame({
    'Number of Clusters': range(2, 11),
    'WCSS': WCSS,
    'Silhouette Score': silhouette_scores,
    'Davies-Bouldin Score': davies_bouldin_scores,
})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nCluster Evaluation Metrics:")
print(cluster_metrics.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(2, 11), WCSS, marker='o')
plt.title('WCSS')
plt.xlabel('Number of Clusters')

plt.subplot(1, 3, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')

plt.subplot(1, 3, 3)
plt.plot(range(2, 11), davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Score')
plt.xlabel('Number of Clusters')


plt.tight_layout()
plt.show()

# Task 6
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

WCSS = []
silhouette_scores = []
davies_bouldin_scores = []

for n_clusters in range(2, 11):
    kmeans_normalized = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_normalized.fit(features_scaled)
    labels = kmeans_normalized.labels_

    WCSS.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, labels))
    davies_bouldin_scores.append(davies_bouldin_score(features, labels))


cluster_metrics = pd.DataFrame({
    'Number of Clusters': range(2, 11),
    'WCSS': WCSS,
    'Silhouette Score': silhouette_scores,
    'Davies-Bouldin Score': davies_bouldin_scores,
})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nCluster Normalized Metrics:")
print(cluster_metrics.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(2, 11), WCSS, marker='o')
plt.title('WCSS')
plt.xlabel('Number of Clusters')

plt.subplot(1, 3, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')

plt.subplot(1, 3, 3)
plt.plot(range(2, 11), davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Score')
plt.xlabel('Number of Clusters')


plt.tight_layout()
plt.show()