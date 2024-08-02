import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Read in results
results = pd.read_csv(r'Models/1_Simulated_data/pysr/results.csv')
results['Operator_complexity'] = 'Low'
results.loc[8:15, 'Operator_complexity'] = 'Medium'
results.loc[16:23, 'Operator_complexity'] = 'High'

# Convert to wide format for RMSE heatmap
rmse_wide = results.pivot(index = 'Operator_complexity', columns = 'Input', values = 'RMSE')

# Define the desired order for rows and columns
row_order = ['High', 'Medium', 'Low']
column_order = ['No_noise_rel_var', 'No_noise_all_var', 'Noise=0.5', 'Noise=1', 'Noise=2', 'Noise=3', 'Noise=4', 'Noise=5']  
rmse_wide = rmse_wide.reindex(index=row_order, columns=column_order)

# Plot heatmap
sns.heatmap(rmse_wide, cmap='viridis', cbar=True, 
            xticklabels=rmse_wide.columns, yticklabels=rmse_wide.index, 
            linecolor='white', linewidths=0.5)  # Add cell lines)
plt.xlabel('Input')
plt.ylabel('Operator Complexity')
plt.title('Heatmap of RMSE by Input and Operator Complexity')
plt.show()


# Convert to wide for correctness heatmap
cor_wide = results.pivot(index = 'Operator_complexity', columns = 'Input', values = 'Correct')

# Define the desired order for rows and columns
cor_wide = cor_wide.reindex(index=row_order, columns=column_order)

# Plot heatmap
sns.heatmap(cor_wide, cmap='viridis', cbar=True, 
            xticklabels=cor_wide.columns, yticklabels=cor_wide.index, 
            linecolor='white', linewidths=0.5)  # Add cell lines)
plt.xlabel('Input Complexity')
plt.ylabel('Operator Complexity')
plt.title('Heatmap of RMSE by Input and Operator Complexity')
plt.show()
