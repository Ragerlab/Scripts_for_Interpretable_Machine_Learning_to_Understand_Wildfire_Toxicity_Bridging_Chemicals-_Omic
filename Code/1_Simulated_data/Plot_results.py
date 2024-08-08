import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Read in results
results_pysr = pd.read_csv(r'Models/1_Simulated_data/pysr/results.csv')
results_gplearn = pd.read_csv(r'Models/1_Simulated_data/gplearn/results.csv')
results_feyn = pd.read_csv(r'Models/1_Simulated_data/feyn/results.csv')

# Putting DataFrames into a dictionary for easier handling
results_dict = {
    'pysr': results_pysr,
    'gplearn': results_gplearn,
    'feyn': results_feyn
}

# Loop to add columns and modify values
for name, df in results_dict.items():
    df['Operator_complexity'] = 'Low'
    df.loc[8:15, 'Operator_complexity'] = 'Medium'
    df.loc[16:23, 'Operator_complexity'] = 'High'
    df['Library'] = name

# Combine into 1 df
results_all = pd.concat(results_dict.values())
results_all['Level'] = results_all['Library'] + results_all['Operator_complexity']


# Convert to wide format for RMSE heatmap
rmse_wide = results_all.pivot(index = 'Level', columns = 'Input', values = 'RMSE')

# Define the desired order for rows and columns
row_order = ['pysrHigh', 'gplearnHigh', 'feynHigh','pysrMedium', 'gplearnMedium', 'feynMedium', 'pysrLow', 'gplearnLow', 'feynLow'] 
column_order = ['No_noise_rel_var', 'No_noise_all_var', 'Noise=0.5', 'Noise=1', 'Noise=2', 'Noise=3', 'Noise=4', 'Noise=5']  
rmse_wide = rmse_wide.reindex(index=row_order, columns=column_order)

# Plot heatmap
sns.heatmap(rmse_wide, cmap='viridis', cbar=True, 
            xticklabels=rmse_wide.columns, yticklabels=rmse_wide.index, 
            linecolor='white', linewidths=0.5)  # Add cell lines)

# Customize with thicker lines every 3 rows
ax = plt.gca()  # get the current axis
for i in range(3, len(row_order), 3):  # start from row index 3 and step by 3
    ax.axhline(i, color='white', lw=2)  # add thicker white line at each 3rd row boundary

plt.xlabel('Input')
plt.ylabel('Operator Complexity')
plt.title('Heatmap of RMSE by Input and Operator Complexity')
plt.show()
plt.savefig(f'images/1_Simulated_data/All_results/rmse_heatmap.png')




# Convert to wide for correctness heatmap
cor_wide = results_all.pivot(index = 'Level', columns = 'Input', values = 'Correct')

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
plt.savefig(f'images/1_Simulated_data/All_results/accuracy_heatmap.png')
