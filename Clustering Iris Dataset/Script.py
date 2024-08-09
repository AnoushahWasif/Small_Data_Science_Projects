import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Standardize the data
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
iris_data['Cluster'] = kmeans.fit_predict(iris_data_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=iris_data.iloc[:, 0], y=iris_data.iloc[:, 1], hue=iris_data['Cluster'], palette='viridis', s=100)
plt.title('Clustering of Iris Dataset')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
