import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

file_path = 'Glass.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['Type'])

# Task 2
k_clusters = len(data['Type'].unique())
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

# Task 3: Visualize clustering results
sns.pairplot(data=data, hue='Cluster', palette='Dark2')
plt.show()

# Task 4
instances_per_cluster = data['Cluster'].value_counts()
print("Instances per cluster:")
print(instances_per_cluster)

cluster_class_distribution = data.groupby(['Cluster', 'Type']).size().unstack(fill_value=0)
print("\nClass distribution in each cluster:")
print(cluster_class_distribution)

# Task 5
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_scores.append(silhouette_score(X, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

cluster_metrics = pd.DataFrame({
    'Number of Clusters': range(2, 11),
    'Silhouette Score': silhouette_scores,
    'Davies-Bouldin Score': davies_bouldin_scores,
    'Calinski-Harabasz Score': calinski_harabasz_scores
})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\nCluster Evaluation Metrics:")
print(cluster_metrics.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot(range(2, 11), davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Score')

plt.subplot(1, 3, 3)
plt.plot(range(2, 11), calinski_harabasz_scores, marker='o')
plt.title('Calinski-Harabasz Score')

plt.tight_layout()
plt.show()

# Task 6
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
data['Cluster_Scaled'] = kmeans_scaled.fit_predict(X_scaled)

# Compare unscaled and scaled clustering results
print("\nComparison of Clustering Results:")
print(data[['Cluster', 'Cluster_Scaled']].head(10))




