import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import seaborn as sns
from sklearn.datasets import make_spd_matrix

# Set random seed for reproducibility
np.random.seed(17)

# Number of data points
n = 50

# Generate random data for 15 chemicals with various distributions and ranges
chem1 = np.random.uniform(1, 10, n)
chem2 = np.random.uniform(2, 5.0, n)
chem3 = np.random.uniform(4, 12, n)
chem4 = np.random.uniform(0.5, 8, n)
chem5 = np.random.normal(loc=3, scale=1, size=n)
chem6 = np.random.normal(loc=5, scale=1.5, size=n)
chem7 = np.random.normal(loc=7, scale=2, size=n)
chem8 = np.random.normal(loc=8, scale=2, size=n)
chem9 = np.random.beta(a=2, b=5, size=n)
chem10 = np.random.beta(a=5, b=2, size=n)
chem11 = np.random.beta(a=3, b=6, size=n)
chem12 = np.random.beta(a=1, b=2, size=n)
chem13 = np.random.lognormal(mean=0, sigma=1, size=n)
chem14 = np.random.lognormal(mean=2, sigma=1, size=n)
chem15 = np.random.lognormal(mean=1, sigma=0.5, size=n)

# # Stack the data into a matrix
data = np.vstack((chem1, chem2, chem3, chem4, chem5, chem6, chem7, chem8, chem9, chem10,
                  chem11, chem12, chem13, chem14, chem15)).T

# Create a correlation matrix with low correlations
corr_matrix = np.eye(15)
for i in range(15):
    for j in range(i + 1, 15):
        corr_value = np.random.uniform(0.2, 0.5) * np.random.choice([-1, 1])
        corr_matrix[i, j] = corr_value
        corr_matrix[j, i] = corr_value

# Use the Cholesky decomposition to ensure the matrix is positive definite
corr_matrix = np.dot(corr_matrix, corr_matrix.T)

# Normalize the diagonal elements to 1 to make it a correlation matrix
D = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
corr_matrix = np.dot(np.dot(D, corr_matrix), D)

# Perform Cholesky decomposition on the correlation matrix
L = np.linalg.cholesky(corr_matrix)

# Apply the correlation structure to the data
correlated_data = data @ L

# # Split the correlated data back into individual chemicals
chem1, chem2, chem3, chem4, chem5, chem6, chem7, chem8, chem9, chem10, \
chem11, chem12, chem13, chem14, chem15 = correlated_data.T

# Calculate the response
response = (0.5 * chem2 * chem6 
            + 5 * (chem15 / chem10))

# Create a DataFrame
sim_dat = pd.DataFrame({
    'chem1': chem1,
    'chem2': chem2,
    'chem3': chem3,
    'chem4': chem4,
    'chem5': chem5, 
    'chem6': chem6,
    'chem7': chem7,
    'chem8': chem8,
    'chem9': chem9,
    'chem10': chem10,
    'chem11': chem11,
    'chem12': chem12,
    'chem13': chem13,
    'chem14': chem14,
    'chem15': chem15,
    'Response': response
})

# Check the correlation matrix to verify correlations
corr_df = sim_dat.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Chemicals')
plt.show()

# Create dataframe only containing variables included in the equation
sim_dat_rel = pd.DataFrame({
    'chem3': chem2,
    'chem5': chem6,
    'chem7': chem10,
    'chem12': chem13, 
    'Response': response
})

# Save simulated dataframe
sim_dat.to_pickle("Data_inputs/1_Simulated_data/sim_dat_all")
sim_dat_rel.to_pickle("Data_inputs/1_Simulated_data/sim_dat_rel")

# Initialize dictionary to hold noisy dfs
sim_noise_dict = {}

# Add noise to complete dataset
for i in range(6):
    # Generage noise from gausian with mean 0, std dev i+1, and n samples 
    if i==0: 
        std = 0.5
    else:
        std = i

    noise = np.random.normal(0, i+1, n)
    response_noisy = response + noise

    # Compare noisy to actual values 
    # plt.scatter(response, response_noisy)
    # plt.xlabel('Actual respsonse')
    # plt.ylabel('Noisy response')
    # plt.title('Noise = mean 0 + std dev' + str(i+1))
    # plt.show()

    # Update dataframe to have noisy response
    sim_noise_temp = sim_dat.assign(Response = response_noisy)
    
    # Append to dictionary
    sim_noise_dict['Noise=' + str(i + 1)] = sim_noise_temp

# Combine all dfs into one dictionary to iterate through
sim_dict = {}
sim_dict['No_noise_all_var'] = sim_dat
sim_dict['No_noise_rel_var'] = sim_dat_rel
sim_dict.update(sim_noise_dict)

# Save simulated dataframe
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'wb') as f:
    pickle.dump(sim_dict, f)
