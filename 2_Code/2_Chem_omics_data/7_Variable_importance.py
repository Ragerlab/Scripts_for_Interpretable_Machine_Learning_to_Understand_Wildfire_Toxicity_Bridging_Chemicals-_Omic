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
from func_timeout import func_timeout, FunctionTimedOut

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

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

# Define datasets to process
datasets = [
    {
        "prefix": "Chem",
        "train_x": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x",
        "train_x_pca": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x_pca",
        "train_x_elastic": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x_elastic",
        "base_hof_directory": "4_Model_results/2_Chemical_measurements/pysr/HOF_all_iterations",
        "results_directory": "4_Model_results/2_Chemical_measurements/pysr",
        "images_directory": "5_Plots/2_Chemical_measurements/pysr"
    },
    {
        "prefix": "Omic",
        "train_x": "3_Data_intermediates/3_Omic_measurements/Omic_train_x",
        "train_x_pca": "3_Data_intermediates/3_Omic_measurements/Omic_train_x_pca",
        "train_x_elastic": "3_Data_intermediates/3_Omic_measurements/Omic_train_x_elastic",
        "base_hof_directory": "4_Model_results/3_Omic_measurements/pysr/HOF_all_iterations",
        "results_directory": "4_Model_results/3_Omic_measurements/pysr",
        "images_directory": "5_Plots/3_Omic_measurements/pysr"
    }
]

# Subdirectories for Full, PCA, and Elastic
subdirectories = ['Full', 'PCA', 'Elastic']

# Process each dataset
for dataset in datasets:
    prefix = dataset["prefix"]
    base_hof_directory = dataset["base_hof_directory"]
    results_directory = dataset["results_directory"]
    images_directory = dataset["images_directory"]

    # Initialize a dictionary to store the concatenated DataFrames for each subdirectory
    hof_dataframes = {}

    # Function to process a subdirectory and return the concatenated DataFrame
    def process_hof_directory(subdirectory):
        hof_directory = os.path.join(base_hof_directory, subdirectory)

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

                # Add a new column for the iteration number and directory (Full, PCA, or Elastic)
                df['Iteration'] = iteration_num
                df['Directory'] = subdirectory  # Track which subdirectory this data came from

                # Append this DataFrame to the list
                hof_dfs.append(df)

        # Concatenate all DataFrames for this subdirectory
        combined_hof_df = pd.concat(hof_dfs, ignore_index=True)

        # Filter based on RMSE (assuming 'loss' is the relevant column)
        combined_hof_df = combined_hof_df[combined_hof_df['loss'] < 19]

        return combined_hof_df

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Process each of the subdirectories 
    for subdirectory in subdirectories:
        combined_hof_df = process_hof_directory(subdirectory)
        hof_dataframes[subdirectory] = combined_hof_df  

    # Load in chemical ranges data 
    train_x = pd.read_pickle(dataset["train_x"])
    train_x_pca = pd.read_pickle(dataset["train_x_pca"])
    # Do not load 'train_x_elastic' for ranges, as per your request

    # Clean the column names
    train_x_clean = clean_column_names(train_x)
    train_x_clean_pca = clean_column_names(train_x_pca)
    # Do not clean 'train_x_elastic' for ranges

    # Merge the DataFrames by row names (indices) for ranges
    merged_df = pd.merge(train_x_clean, train_x_clean_pca, left_index=True, right_index=True)

    # Calculate ranges for each chemical (min, max)
    chemical_ranges = {}
    for col in merged_df.columns:
        min_value = merged_df[col].min()
        max_value = merged_df[col].max()
        chemical_ranges[col] = (min_value, max_value)

    # Define keys for loop
    keys = list(hof_dataframes.keys())

    # Iterate through relevant DataFrames
    for idx in range(len(keys)): 
        # Subset to the relevant DataFrame
        key = keys[idx]
        combined_hof_df = hof_dataframes[key]
        print(f"Processing subdirectory: {key} for dataset {prefix}")

        # Get all chemical names from the equations
        chems = set()

        # Iterate through each row to extract chemicals
        for equation in combined_hof_df['equation']:
            # Convert the string equation to a sympy expression
            expr = sp.sympify(equation)

            # Get the free symbols 
            chems.update(expr.free_symbols)


        # Convert the set to a list
        chems = list(chems)

        # Initialize an empty DataFrame to hold the final results
        final_results_df = pd.DataFrame(columns=["chem", "equation", "parital derivative w/ respect to chem", "integrated_derivative"])

        # Iterate through each chemical
        for j in range(len(chems)):  
            print(f'j_{j}')

            # Chemical of interest
            chem = chems[j]

            # Subset HOF to only equations containing this chemical
            subset_df = combined_hof_df[combined_hof_df['equation'].str.contains(rf'\b{chem}\b', na=False)]

            # Get unique equations so not performing repeat calculations
            uniq_eqs = subset_df['equation'].unique()

            # Initialize df to hold results for subset
            results_df = pd.DataFrame(columns=["chem", "equation", "parital derivative w/ respect to chem", "integrated_derivative"])

            # Iterate through each row in the subset DataFrame
            for k in range(len(uniq_eqs)): 
                print(f'k_{k}')

                try:
                    # Get the equation 
                    equation_str = uniq_eqs[k]

                    # Convert the equation string to a sympy expression
                    equation_sympy = sp.sympify(equation_str)

                    # Compute the partial derivative of the equation with respect to the chemical
                    partial_derivative = sp.diff(equation_sympy, chem)

                    # Identify all variables in the equation
                    all_symbols = list(partial_derivative.free_symbols)

                    # Ensure that 'chem' is included in the integration variables
                    if chem not in all_symbols:
                        all_symbols.append(chem)

                    # Define the ranges for all chemicals
                    ranges = []
                    for sym in all_symbols:
                        sym_str = str(sym)  # Convert sympy symbol to string to match the column names
                        if sym_str in chemical_ranges:
                            ranges.append(chemical_ranges.get(sym_str)) 
                        else:
                            # If the range for a symbol is not found, skip integration
                            raise ValueError(f"Range for symbol {sym_str} not found.")

                    # Integrate the partial derivative over all variables 
                    integrated_derivative = integrate_over_all_variables(partial_derivative, all_symbols, ranges)
                
                except Exception as e:
                    # If there's an issue, log the error message in the 'derivative' column
                    partial_derivative = f"Error"
                    integrated_derivative = "Error"

                # Ensure the equation is stored as a string
                results_df.loc[len(results_df)] = {
                    "chem": chem,
                    "equation": equation_str,  # Convert sympy expression to string
                    "parital derivative w/ respect to chem": partial_derivative,
                    "integrated_derivative": integrated_derivative,
                }

            # Merge with subset_df
            results_subset = pd.merge(subset_df, results_df, how='left', on='equation')

            # Concatenate the results_subset into the final_results_df
            final_results_df = pd.concat([final_results_df, results_subset], ignore_index=True)
            final_results_df = final_results_df.iloc[:, :7]


        # Save results for each subdirectory
        os.makedirs(results_directory, exist_ok=True)
        results_file_name = os.path.join(results_directory, f'partial_deriv_{key}.csv')
        final_results_df.to_csv(results_file_name, index=False)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculate variable importance 
    for idx in range(len(keys)):  
        # Subset to the relevant DataFrame
        key = keys[idx]

        # Get integrated info
        file_path = os.path.join(results_directory, f'partial_deriv_{key}.csv')
        results_df = pd.read_csv(file_path)

        # Calculate importance score for each variable
        var_importance_list = []

        # Iterate through each unique chemical
        chems = results_df['chem'].unique()
        for j in range(len(chems)):  
            # Chemical of interest
            chem = chems[j]

            # Get rows corresponding to the current chemical
            chem_rows = results_df[results_df['chem'] == chem]
            chem_rows = chem_rows.drop_duplicates()

            # Sum the 'integrated_derivative' for the current chemical
            sum_integrated_derivative = chem_rows['integrated_derivative'].sum()

            # Get chemical concentrations
            chem = str(chem)
            if chem in chemical_ranges:
                min_value, max_value = chemical_ranges[chem]
                range_value = max_value - min_value

                # Calculate the var_importance
                var_importance = sum_integrated_derivative / range_value
            else:
                # If range is not available, set var_importance to NaN
                var_importance = np.nan

            # Append the chemical name and its var_importance to the list
            var_importance_list.append([chem, var_importance])

        # Create a new DataFrame for var_importance
        var_importance_df = pd.DataFrame(var_importance_list, columns=['chem', 'var_importance'])
        var_importance_df['chem'] = var_importance_df['chem'].astype(str)

        # Sort the DataFrame by 'var_importance' in decreasing order
        var_importance_df = var_importance_df.sort_values(by='var_importance', ascending=False)

        # Save results for each subdirectory
        var_importance_name = os.path.join(results_directory, f'variable_importance_{key}.csv')
        var_importance_df.to_csv(var_importance_name, index=False)

        # Sort the DataFrame by the absolute value of 'var_importance' in decreasing order and subset the top 15
        top_15_abs = var_importance_df.reindex(var_importance_df['var_importance'].abs().sort_values(ascending=False).index).head(15)

        # Sort the subset based on the original 'var_importance' values to maintain the original order
        top_15_df = top_15_abs.sort_values(by='var_importance', ascending=False if top_15_abs['var_importance'].iloc[0] > 0 else True)

        # Create directories for images
        os.makedirs(images_directory, exist_ok=True)

        # Create a bar plot for each subdirectory
        plt.figure(figsize=(10, 6))
        plt.bar(top_15_df['chem'], top_15_df['var_importance'])
        plt.xlabel('Chemical')
        plt.ylabel('Variable Importance')
        plt.title(f'Variable Importance by Chemical ({key}) - {prefix}')
        plt.xticks(rotation=90)  # Rotate chemical names for better readability
        plt.tight_layout()
        plt.show()
        # Save the plot
        plt.savefig(os.path.join(images_directory, f'var_importance_{key}.png'))
