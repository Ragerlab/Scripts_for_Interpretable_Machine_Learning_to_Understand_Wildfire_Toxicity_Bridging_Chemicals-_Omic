import os
import pandas as pd
import pickle
import sympy as sp
import re
from scipy.integrate import nquad, IntegrationWarning
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# Function to perform numerical integration over all variables using nquad, focusing on the real part
def integrate_over_all_variables(partial_derivative, all_symbols, ranges):
    try:
        # Extract only the real part of the partial_derivative
        real_part = sp.re(partial_derivative)

        # Convert the SymPy expression (real part) into a numerical function using lambdify
        func = sp.lambdify(all_symbols, real_part, 'numpy')

        # Define the integrand function for nquad
        def integrand(*args):
            try:
                result = func(*args)
                # Convert result to float, in case it's an array
                result = float(result)
                if np.isnan(result) or np.isinf(result):
                    return 0
                else:
                    return result
            except Exception:
                # If an exception occurs in the integrand, return 0
                return 0

        # Perform integration without timeout
        with warnings.catch_warnings():
            # Convert IntegrationWarnings to exceptions
            warnings.simplefilter("error", IntegrationWarning)
            # Suppress runtime warnings (e.g., overflow, divide by zero)
            warnings.simplefilter("ignore", RuntimeWarning)
            result, _ = nquad(integrand, ranges)

        return result

    except Exception as e:
        print(f"Error during numerical integration setup: {str(e)}")
        return 0

    
# Directory where the HOF files are stored
hof_directory = r"Models/3_Omic_measurements/HOF_all_iterations"

 # Initialize an empty list to store DataFrames for this subdirectory
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

# Concatenate all DataFrames for this subdirectory
combined_hof_df = pd.concat(hof_dfs, ignore_index=True)

# Filter based on RMSE (assuming 'loss' is the relevant column)
combined_hof_df = combined_hof_df[combined_hof_df['loss'] < 15]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load in geneical ranges data 
injury_df = pd.read_pickle("Data_inputs/3_Omic_measurements/dat_deg")

# Remove the 'Injury_Protein' column
injury_df_cleaned = injury_df.drop(columns=['Injury_Protein'])

 # Get all gene names from the equations
genes = set()

# Iterate through each row to extract genes
for equation in combined_hof_df['equation']:
        # Convert the string equation to a sympy expression
        expr = sp.sympify(equation)

        # Get the free symbols 
        genes.update(expr.free_symbols)

# Convert the set to a list
genes = list(genes)

# Calculate ranges for each geneical (min, max)
gene_ranges = {}
for col, gene_name in zip(injury_df_cleaned.columns, genes):
    min_value = injury_df_cleaned[col].min()
    max_value = injury_df_cleaned[col].max()
    gene_ranges[str(gene_name)] = (min_value, max_value)

# Initialize an empty DataFrame to hold the final results
final_results_df = pd.DataFrame(columns=["gene", "equation", "partial derivative w/ respect to gene", "integrated_derivative", "direction"])

# Iterate through each gene
for j in range(len(genes)):  
    print(f'j_{j}')

    # Gene of interest
    gene = genes[j]

    # Subset HOF to only equations containing this gene
    subset_df = combined_hof_df[combined_hof_df['equation'].str.contains(rf'\b{gene}\b', na=False)]

    # Get unique equations to avoid repeat calculations
    uniq_eqs = subset_df['equation'].unique()

    # Initialize DataFrame to hold results for the subset
    results_df = pd.DataFrame(columns=["gene", "equation", "partial derivative w/ respect to gene", "integrated_derivative", "direction"])

    # Iterate through each unique equation in the subset DataFrame
    for k in range(len(uniq_eqs)): 
        print(f'k_{k}')

        try:
            # Get the equation 
            equation_str = uniq_eqs[k]

            # Convert the equation string to a sympy expression using the symbol dictionary
            equation_sympy = sp.sympify(equation_str)

            # Compute the partial derivative of the equation with respect to the gene
            partial_derivative = sp.diff(equation_sympy, gene)

            # Identify all variables in the equation
            all_symbols = list(partial_derivative.free_symbols)

            # Convert the symbols to strings
            all_symbols = [str(sym) for sym in all_symbols]

            # Ensure that 'gene' is included in the integration variables
            if gene not in all_symbols:
                new_gene = str(sp.sympify(gene))
                all_symbols.append(new_gene)

            # Define the ranges for all variables
            ranges = []
            for sym in all_symbols:
                ranges.append(gene_ranges.get(sym)) 

            # Integrate the partial derivative over all variables 
            integrated_derivative = integrate_over_all_variables(partial_derivative, all_symbols, ranges)

            # Determine the direction based on the integrated_derivative
            if integrated_derivative > 0:
                direction = 1
            elif integrated_derivative < 0:
                direction = -1
            else:
                direction = 0
        
        except Exception as e:
            # If there's an issue, log the error message
            partial_derivative = "Error"
            integrated_derivative = "Error"
            direction = "Error"

        # Add the results to the DataFrame
        results_df.loc[len(results_df)] = {
            "gene": gene,
            "equation": equation_str,
            "partial derivative w/ respect to gene": partial_derivative,
            "integrated_derivative": integrated_derivative,
            "direction": direction
        }

    # Merge with subset_df
    results_subset = pd.merge(subset_df, results_df, how='left', on='equation')

    # Concatenate the results_subset into the final_results_df
    final_results_df = pd.concat([final_results_df, results_subset], ignore_index=True)
    final_results_df = final_results_df.iloc[:, :7]

 # Save results 
results_file_name = f'Models/3_Omic_Measurements/partial_deriv.csv'
final_results_df.to_csv(results_file_name, index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate variable importance 
file_path = f'Models/3_Omic_Measurements/partial_deriv.csv'
results_df = pd.read_csv(file_path)

# Calculate importance score for each variable
var_importance_list = []

# Iterate through each unique gene
genes = results_df['gene'].unique()
for j in range(len(genes)):  
    # gene of interest
    gene = genes[j]

    # Get rows corresponding to the current gene
    gene_rows = results_df[results_df['gene'] == gene]

    # Sum the 'integrated_derivative' for the current gene
    sum_integrated_derivative = gene_rows['direction'].sum()
    # sum_integrated_derivative = gene_rows['integrated_derivative'].sum()

    # Get gene concentrations
    min_value, max_value = gene_ranges[gene]
    range_value = max_value - min_value

    # Calculate the var_importance
    var_importance = sum_integrated_derivative #* (len(gene_rows) / len(combined_hof_df)) * range_value

    # Append the gene name and its var_importance to the list
    var_importance_list.append([gene, var_importance])

# Create a new DataFrame for var_importance
var_importance_df = pd.DataFrame(var_importance_list, columns=['gene', 'var_importance'])
var_importance_df['gene'] = var_importance_df['gene'].astype(str)

# Sort the DataFrame by 'var_importance' in decreasing order
var_importance_df = var_importance_df.sort_values(by='var_importance', ascending=False)

# Create a bar plot for each subdirectory
plt.figure(figsize=(10, 6))
plt.bar(var_importance_df['gene'], var_importance_df['var_importance'])
plt.xlabel('gene')
plt.ylabel('Variable Importance')
plt.title(f'Variable Importance by gene')
plt.xticks(rotation=90)  # Rotate geneical names for better readability
plt.tight_layout()
# Save the plot for each subdirectory
plt.savefig(f'Images/3_Omic_measurements/var_importance.png')


