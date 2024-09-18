import os
import pandas as pd
import pickle
import sympy as sp
import re
from scipy.integrate import quad
from scipy.integrate import nquad
import numpy as np
import matplotlib.pyplot as plt
import math

# Define all functions 

# Define function to clean up names for pysr
def clean_column_names(df):
        new_columns = []
        for col in df.columns:
            if col == 'S':
                new_columns.append('Sulphur')
            elif col == 'Si':
                new_columns.append('Silicon')
            else:
                cleaned_col = re.sub(r'\W+', '', col)
                cleaned_col = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', cleaned_col)
                cleaned_col = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', cleaned_col)
                if cleaned_col[0].isdigit():
                    cleaned_col = 'var' + cleaned_col
                    
                new_columns.append(cleaned_col)
        df.columns = new_columns
        return df

# Function to perform numerical integration over all variables using nquad
def integrate_over_all_variables(partial_derivative, all_symbols, ranges):
    try:
        # Check if the partial_derivative is a constant (no symbols)
        if not partial_derivative.free_symbols:
            # If it's a constant, return the constant value multiplied by the volume of the integration ranges
            constant_value = float(partial_derivative)
            volume = np.prod([upper - lower for lower, upper in ranges])
            return constant_value * volume

        # Convert the SymPy expression into a numerical function using lambdify
        func = sp.lambdify(all_symbols, partial_derivative, 'numpy')

        # Perform multidimensional integration using nquad
        result, _ = nquad(func, ranges)

        return result

    except Exception as e:
        return f"Error during numerical integration: {str(e)}"

# Directory where the HOF files are stored
hof_directory = r"Models/2_Chemical_measurements/pysr/HOF_all_iterations"

# Initialize an empty list to store DataFrames
hof_dfs = []

# Iterate through each file in the HOF directory
for file_name in os.listdir(hof_directory):
    if file_name.endswith(".csv"):
        # Extract iteration number from file name
        iteration_num = int(file_name.split('_')[-1].split('.')[0])

        # Read the CSV file into a DataFrame
        file_path = os.path.join(hof_directory, file_name)
        df = pd.read_csv(file_path)

        # Add a new column for the iteration number
        df['Iteration'] = iteration_num

        # Append this DataFrame to the list
        hof_dfs.append(df)

# Concatenate all DataFrames into one
combined_hof_df = pd.concat(hof_dfs, ignore_index=True)
# combined_hof_df = combined_hof_df.loc[:16,]

# Filter based on RMSE
combined_hof_df = combined_hof_df[combined_hof_df['loss'] < 25]

# Save 
file_name = f'Models/2_Chemical_measurements/pysr/combined_hof.csv'
combined_hof_df.to_csv(file_name, index=False)

# Get all chemical names
# Initialize an empty set to store unique chemicals (free symbols)
chems = set()

# Iterate through each row in the 'equation' column
for equation in combined_hof_df['equation']:
        # Convert the string equation to a sympy expression
        expr = sp.sympify(equation)
        
        # Get the free symbols (chemicals) from the expression and update the set
        chems.update(expr.free_symbols)

# Convert the set to a list first
chems = list(chems)

# Load in chemical ranges data 
injury_df = pd.read_pickle("Data_inputs/2_Chemical_measurements/injury_df")

# Remove the 'Injury_Protein' column
injury_df_cleaned = injury_df.drop(columns=['Injury_Protein'])

# Clean the column names using the provided function
injury_df_cleaned = clean_column_names(injury_df_cleaned)

# Calculate ranges for each chemical (min, max)
chemical_ranges = {}
for col in injury_df_cleaned.columns:
    min_value = injury_df_cleaned[col].min()
    max_value = injury_df_cleaned[col].max()
    chemical_ranges[col] = (min_value, max_value)

# Initialize df to hold results 
results_df = pd.DataFrame(columns=["chem", "sympy_equation", "parital derivative w/ respect to chem", "integrated_derivative", "direction"])

# Iterate through each chemical
for i in range(len(chems)):
    print(i)
    # Chemical of interest
    chem = chems[i]

    # Subset HOF to only equations containing this chemical
    subset_df = combined_hof_df[combined_hof_df['equation'].str.contains(rf'\b{chem}\b', na=False)]

    # Iterate through each row in the subset DataFrame
    for j in range(len(subset_df)):
        try:
            # Get the equation string from the j-th row of the subset DataFrame
            equation_str = subset_df.iloc[j]['equation']

            # Convert the equation string to a sympy expression
            equation_sympy = sp.sympify(equation_str)

            # Compute the partial derivative of the equation with respect to the chemical
            partial_derivative = sp.diff(equation_sympy, chem)

            # Identify all variables in the equation, including chem_symbol
            all_symbols = list(partial_derivative.free_symbols)

            # Define the ranges for all variables based on the chemical_ranges dictionary
            ranges = []
            for sym in all_symbols:
                sym_str = str(sym)  # Convert sympy symbol to string to match the column names
                ranges.append(chemical_ranges.get(sym_str)) 

            # Integrate the partial derivative over all variables (including chem_symbol)
            integrated_derivative = integrate_over_all_variables(partial_derivative, all_symbols, ranges)

            # Determine the direction based on the integrated_derivative
            if integrated_derivative > 0:
                direction = "positive"
            elif integrated_derivative < 0:
                direction = "negative"
            else:
                direction = "neutral"
            
        except Exception as e:
            # If there's an issue, log the error message in the 'derivative' column
            partial_derivative = f"Error"
            integrated_derivative = "Error"
            direction = "Error"

        results_df.loc[len(results_df)] = {
            "chem": chem,
            "sympy_equation": equation_sympy,
            "parital derivative w/ respect to chem": partial_derivative,
            "integrated_derivative": integrated_derivative,
            "direction": direction
        }
        
# Save results
file_name = f'Models/2_Chemical_measurements/pysr/partial_deriv.csv'
results_df.to_csv(file_name, index=False)

# Calculate importance score for each variable
# Create an empty list to store the chemical names and their var_importance
var_importance_list = []

# Iterate through each unique chemical
for chem in results_df['chem'].unique():
    # Get rows corresponding to the current chemical
    chem_rows = results_df[results_df['chem'] == chem]
    
    # Sum the 'integrated_derivative' for the current chemical
    sum_integrated_derivative = chem_rows['integrated_derivative'].sum()
    
    # Calculate the var_importance
    var_importance = abs(sum_integrated_derivative) * (len(chem_rows) / len(combined_hof_df))
    
    # Append the chemical name and its var_importance to the list
    var_importance_list.append([chem, var_importance])

# Create a new DataFrame from the list
var_importance_df = pd.DataFrame(var_importance_list, columns=['chem', 'var_importance'])
var_importance_df['chem'] = var_importance_df['chem'].astype(str)

# Sort the DataFrame by 'var_importance' in decreasing order
var_importance_df = var_importance_df.sort_values(by='var_importance', ascending=False)

# Apply logarithm transformation to the 'var_importance' column using numpy.log
var_importance_df['log_var_importance'] = np.log(var_importance_df['var_importance'])

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(var_importance_df['chem'], var_importance_df['log_var_importance'])
plt.xlabel('Chemical')
plt.ylabel('Log of Variable Importance')
plt.title('Log of Variable Importance by Chemical')
plt.xticks(rotation=90)  # Rotate chemical names for better readability
plt.tight_layout()

# Display the plot
plt.show()