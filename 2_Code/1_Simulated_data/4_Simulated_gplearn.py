from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import root_mean_squared_error
import sympy as sp
import time
import os

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Load in data
with open('3_Data_intermediates/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)    

# Define potential operator inputs
ops = [('add', 'sub'), ('add', 'sub', 'mul', 'div'), ('add', 'sub', 'mul', 'div', 'sqrt', 'log')]
operators= pd.DataFrame({
    'operators': ops,
})
operators.index = ['low', 'med', 'high']

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Input", "Operators", "RMSE", "Equation"])

# Intialize row counting variable for results table
row = 0

# Start timer 
start_time = time.time()

# Iterate through operator options
for i in range(len(operators.index)):
    print(i)

    # Subset to relevant operators
    op_temp = operators.iloc[i, operators.columns.get_loc('operators')]

    # Iterate through different simulated inputs
    for j in range(len(sim_dict)):

        # Subset to one df
        keys = list(sim_dict.keys())
        key = keys[j]
        data = sim_dict[key]
        
        # Define input and output 
        x = data.drop('Response', axis = 1)
        y = data['Response']

        # Note column names to later fix gplearn output
        column_names = x.columns.tolist()
        column_mapping = {f'X{i}': name for i, name in enumerate(column_names)}

        # Set model parameters - https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-regressor
        est_gp = SymbolicRegressor(population_size=1000,
                               generations=500,
                               p_crossover=0.9,
                               p_hoist_mutation = 0.05,
                               max_samples = 0.3,
                               metric='rmse',
                               function_set=op_temp,
                               warm_start=True,
                               parsimony_coefficient= 0.5, # cfhanged from 0.25
                               feature_names=x.columns.tolist(), 
                               verbose = 1,
                               random_state= 17)
        
        # Fit model
        mod = est_gp.fit(x, y)

        # Get top 10 models
        sorted_programs = sorted(est_gp._programs[-1], key=lambda program: program.fitness_, reverse=True)
        equations = sorted_programs[:10]

        sympy_equations = []
        for program in equations:
            # Extract the string representation of the program
            equation_str = program.__str__()

            # Convert the string to a sympy expression
            equation = sp.sympify(equation_str)
            sympy_equations.append(str(equation))

        # Convert the equations to a DataFrame
        df_equations = pd.DataFrame(sympy_equations, columns=['Equation'])

        # Save the DataFrame to a CSV file
        file_name = f'4_Model_results/1_Simulated_data/gplearn/{list(operators.index)[i]}_{list(sim_dict.keys())[j]}_HOF.csv'
        df_equations.to_csv(file_name, index=False)

        # Calc RMSE
        y_pred = mod.predict(x)
        rmse = root_mean_squared_error(y,y_pred)
        print(f"Training RMSE: {rmse:.2f}")

        # Save results
        results_df.loc[row]=[list(sim_dict.keys())[j], op_temp, rmse, sympy_equations[1]]
        row = row + 1

# End timer and calculate time taken 
end_time = time.time()
time_taken = end_time - start_time

results_df['time'] = time_taken

# Save to csv
results_df
file_name = f'4_Model_results/1_Simulated_data/gplearn/results_gplearn.csv'
results_df.to_csv(file_name, index=False)

