import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle

# Set random seed for reproducibility
np.random.seed(17)

# Number of data points
n = 50

# Generate random data for 15 chemicals with various distributions and ranges
chem1 = np.random.uniform(1, 10, n)
chem2 = np.random.uniform(2, 5.0, n)
chem3 = np.random.uniform(4, 12, n)
chem4 = np.random.uniform(0.5, 6, n)
chem5 = np.random.uniform(6, 10, n)
chem6 = np.random.uniform(1, 7, n)
chem7 = np.random.normal(loc=3, scale=1, size=n)
chem8 = np.random.normal(loc=5, scale=1.5, size=n)
chem9 = np.random.normal(loc=7, scale=2, size=n)
chem10 = np.random.beta(a=2, b=5, size=n)
chem11 = np.random.beta(a=5, b=2, size=n)
chem12 = np.random.normal(loc=8, scale=2, size=n)
chem13 = np.random.normal(loc=4, scale=1.5, size=n)
chem14 = np.random.lognormal(mean=0, sigma=1, size=n)
chem15 = np.random.lognormal(mean=1, sigma=0.5, size=n)

# Calculate the response
response = (0.5 * chem3 
            + chem3 * chem5 
            + 5*(chem7 / chem12))


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

# Create dataframe only containing variables included in the equation
sim_dat_rel = pd.DataFrame({
    'chem3': chem3,
    'chem5': chem5,
    'chem7': chem7,
    'chem12': chem12, 
    'Response': response
})

# Save simulated dataframe
sim_dat.to_pickle("Data_inputs/1_Simulated_data/sim_dat_all")
sim_dat_rel.to_pickle("Data_inputs/1_Simulated_data/sim_dat_rel")

# Initialize dictionary to hold noisy dfs
sim_noise_dict = {}

# Add noise to complete dataset
for i in range(5):
    # Generage noise from gausian with mean 0, std dev i+1, and n samples 
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
    sim_noise_dict['Noise=' + str(i)] = sim_noise_temp

# Combine all dfs into one dictionary to iterate through
sim_dict = sim_noise_dict
sim_dict['No_noise_all_var'] = sim_dat
sim_dict['No_noise_rel_var'] = sim_dat_rel

# Save simulated dataframe
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'wb') as f:
    pickle.dump(sim_dict, f)
