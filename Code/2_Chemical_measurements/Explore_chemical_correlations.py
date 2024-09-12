import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import statsmodels.stats.multitest as smm


# Read in data 
injury_df = pd.read_pickle("Data_inputs/2_Chemical_measurements/injury_df")

# Drop the 'Injury_Protein' column
data = injury_df.drop(columns=['Injury_Protein'])

# Initialize matrices to store the p-values and correlation coefficients
cols = data.columns
p_values = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
correlation_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)

# Calculate pairwise Pearson correlations and p-values
for i in range(len(cols)):
    for j in range(i, len(cols)):
        if i != j:
            corr, p_value = pearsonr(data[cols[i]], data[cols[j]])
            correlation_matrix.loc[cols[i], cols[j]] = corr
            correlation_matrix.loc[cols[j], cols[i]] = corr
            p_values.loc[cols[i], cols[j]] = p_value
            p_values.loc[cols[j], cols[i]] = p_value
        else:
            correlation_matrix.loc[cols[i], cols[j]] = 1  # Perfect correlation with itself
            p_values.loc[cols[i], cols[j]] = 0  # No p-value for correlation with itself

# Flatten the p-values matrix for FDR correction
p_values_flat = p_values.values.flatten()

# Apply FDR correction using the Benjamini-Hochberg procedure
rejected, p_values_corrected = smm.fdrcorrection(p_values_flat, alpha=0.05)

# Reshape the corrected p-values back into a DataFrame with the same shape as the original p-values matrix
p_values_corrected = p_values_corrected.reshape(p_values.shape)
p_values_corrected_df = pd.DataFrame(p_values_corrected, index=p_values.index, columns=p_values.columns)

# Create a mask for non-significant correlations (p-value > threshold)
significance_mask = p_values.applymap(lambda x: x <= 0.05)

# Create a heatmap where only significant correlations are colored
#correlation_matrix = correlation_matrix[["Si", "P", "Isoeugenol"]]
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=~significance_mask, annot=False, fmt='.2f', cmap='coolwarm', cbar=True, linewidths=0.5)
#sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', cbar=True, linewidths=0.5)
plt.title('Correlation Matrix (Significant Correlations Highlighted)')
plt.show()