import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data
train_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x")
test_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x")

# Transpose the df so that chemicals are in rows
train_x_T = train_x.T

# Standardize the data
scaler = StandardScaler()
train_x_T_scaled = scaler.fit_transform(train_x_T)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hierarchical clustering

# Compute the distance between chemicals 
distance_matrix = pdist(train_x_T_scaled, metric='euclidean')

# Perform hierarchical clustering
Z = linkage(distance_matrix, method='complete')

# Plot
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=train_x_T.index, leaf_rotation=90)
plt.title("Hierarchical Clustering of Chemicals")
plt.xlabel("Chemicals")
plt.ylabel("Euclidean Distance")
plt.show()

# Cut the dendrogram to form flat clusters, using a distance threshold
distance_threshold = 3  # Example threshold, adjust as needed
clusters = fcluster(Z, t=distance_threshold, criterion='distance')

# Add the cluster labels to your chemicals DataFrame
chemical_clusters = pd.DataFrame({'Chemical': train_x_T.index, 'Cluster': clusters})

# Display the cluster assignment
print(chemical_clusters)

# Optionally, you can group by clusters and see which chemicals are in each cluster
grouped_clusters = chemical_clusters.groupby('Cluster')['Chemical'].apply(list)
print(grouped_clusters)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# K-means
# Step 1: Elbow Method to determine optimal number of clusters
inertia = []
k_range = range(1, 11)  # Check for cluster numbers between 1 and 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_x_T_scaled)
    inertia.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.show()

# Run Kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
train_x_T['Cluster'] = kmeans.fit_predict(train_x_T_scaled)

# Add the cluster labels to your chemicals DataFrame
chemical_clusters = pd.DataFrame({'Chemical': train_x_T.index, 'Cluster': train_x_T['Cluster']})

# Display the cluster assignment
print(chemical_clusters)

# Optionally, you can group by clusters and see which chemicals are in each cluster
grouped_clusters = chemical_clusters.groupby('Cluster')['Chemical'].apply(list)
print(grouped_clusters)