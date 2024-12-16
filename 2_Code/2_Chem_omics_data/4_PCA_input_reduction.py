import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Define paths for datasets
datasets = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_x": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x",
        "test_x": "3_Data_intermediates/2_Chemical_measurements/Chem_test_x"
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_x": "3_Data_intermediates/3_Omic_measurements/Omic_train_x",
        "test_x": "3_Data_intermediates/3_Omic_measurements/Omic_test_x"
    }
]

# Loop through datasets using indices
for i in range(len(datasets)):
    dataset = datasets[i]
    print(f"Processing {dataset['prefix']} dataset...")

    # Load data
    train_x = pd.read_pickle(dataset["train_x"])
    test_x = pd.read_pickle(dataset["test_x"])

    # Standardize the training data
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)

    # Create PCA object and fit it to the training data
    pca = PCA(n_components=10)
    pca.fit(train_x_scaled)

    # Scree plot
    output_dir = f"5_Plots/{dataset['path']}/pca"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1))
    plt.grid(True)
    plt.show()
    plt.savefig(f'{output_dir}/scree_plot.png')
    plt.close()

    # Transform training data to principal components
    train_pca_scores = pca.transform(train_x_scaled)

    # Scores plot
    plt.figure(figsize=(10, 6))
    plt.scatter(train_pca_scores[:, 0], train_pca_scores[:, 1], alpha=0.8)
    for j, label in enumerate(train_x.index):
        plt.annotate(label, (train_pca_scores[j, 0], train_pca_scores[j, 1]))
    plt.title('Scores Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(f'{output_dir}/scores_plot.png')
    plt.close()

    # Retain the top 4 principal components for training data
    train_x_pca = pd.DataFrame(train_pca_scores[:, 0:4], columns=['PC1', 'PC2', 'PC3', 'PC4'], index=train_x.index)

    # Standardize the test data using the same scaler
    test_x_scaled = scaler.transform(test_x)

    # Transform test data to principal components
    test_pca_scores = pca.transform(test_x_scaled)

    # Retain the top 4 principal components for test data
    test_x_pca = pd.DataFrame(test_pca_scores[:, 0:4], columns=['PC1', 'PC2', 'PC3', 'PC4'], index=test_x.index)

    # Save reduced DataFrames
    output_data_dir = f"3_Data_intermediates/{dataset['path']}"
    os.makedirs(output_data_dir, exist_ok=True)
    train_x_pca.to_pickle(f"{output_data_dir}/{dataset['prefix']}_train_x_pca")
    test_x_pca.to_pickle(f"{output_data_dir}/{dataset['prefix']}_test_x_pca")

    print(f"Finished processing {dataset['prefix']} dataset.\n")
