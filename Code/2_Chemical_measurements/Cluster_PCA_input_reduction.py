import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

# Load the data
train_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x")
test_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x")

#Transpose the DataFrame so that chemicals are in rows
train_x_T = train_x.T

# Standardize the data
scaler = StandardScaler()
train_x_T_scaled = scaler.fit_transform(train_x_T)

# Compute the Euclidean distance between chemicals (on scaled data)
distance_matrix = pdist(train_x_T_scaled, metric='euclidean')

# Perform hierarchical clustering
Z = linkage(distance_matrix, method='complete')

# Step 4: Plot the dendrogram
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



# Reduce dimensionality using PCA
# Subset data and standardize values
scaler = StandardScaler()
pca_sub_scaled = scaler.fit_transform(train_x)

# Create PCA object
pca = PCA(n_components = 10)

# Fit PCA to data excluding protein injury
pca.fit(pca_sub_scaled)

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()

# Transform data to principal components
pca_scores = pca.transform(pca_sub_scaled)

# Scores plot 
plt.figure(figsize=(10, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.8)
# Add labels based on row names in pca_sub_unique
for i, label in enumerate(train_x.index):
    plt.annotate(label, (pca_scores[i, 0], pca_scores[i, 1]))
plt.title('Scores Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Pull out the top 4 components
pcs_retain = pca_scores[:,0:4]
train_x_pca = pd.DataFrame(pcs_retain, columns=['PC1', 'PC2', 'PC3', 'PC4']) # Set column names

# Set the row names
train_x_pca.index = train_x.index

# Standardize the test data using the same scaler fitted on the training data
test_x_scaled = scaler.transform(test_x)

# Transform test data to principal components using the same PCA model
test_pca_scores = pca.transform(test_x_scaled)

# Pull out the top 4 components for test data
pcs_retain_test = test_pca_scores[:, 0:4]
test_x_pca = pd.DataFrame(pcs_retain_test, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Set the row names for test data
test_x_pca.index = test_x.index

# Save reduced dataframes
train_x_pca.to_pickle("Data_inputs/2_Chemical_measurements/train_x_pca")
test_x_pca.to_pickle("Data_inputs/2_Chemical_measurements/test_x_pca")
