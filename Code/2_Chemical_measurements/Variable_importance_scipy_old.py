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

        # Function to perform integration with a timeout
        def perform_integration():
            with warnings.catch_warnings():
                # Convert IntegrationWarnings to exceptions
                warnings.simplefilter("error", IntegrationWarning)
                # Suppress runtime warnings (e.g., overflow, divide by zero)
                warnings.simplefilter("ignore", RuntimeWarning)
                return nquad(integrand, ranges)

        # Perform numerical integration using func_timeout
        try:
            # Set the timeout duration (in seconds)
            TIMEOUT_DURATION = 5  # Adjust this value as needed

            # Attempt to perform the integration within the timeout
            result, _ = func_timeout(TIMEOUT_DURATION, perform_integration)

            return result

        except FunctionTimedOut:
            print("Integration took too long. Setting integral to 0.")
            return 0
        except Exception as e:
            print(f"Error during integration: {e}")
            return 0

    except Exception as e:
        print(f"Error during numerical integration setup: {str(e)}")
        return 0
    
# Directory where the HOF files are stored
base_hof_directory = r"Models/2_Chemical_measurements/pysr/HOF_all_iterations"

# Subdirectories for Full, PCA, and Lasso
subdirectories = ['Full', 'Lasso']

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

            # Add a new column for the iteration number and directory (Full, PCA, or Lasso)
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
injury_df = pd.read_pickle("Data_inputs/2_Chemical_measurements/injury_df")

# Remove the 'Injury_Protein' column
injury_df_cleaned = injury_df.drop(columns=['Injury_Protein'])

# Clean the column names
injury_df_cleaned = clean_column_names(injury_df_cleaned)

# Calculate ranges for each chemical (min, max)
chemical_ranges = {}
for col in injury_df_cleaned.columns:
    min_value = injury_df_cleaned[col].min()
    max_value = injury_df_cleaned[col].max()
    chemical_ranges[col] = (min_value, max_value)

# Define keys for loop
keys = list(hof_dataframes.keys())

# Iterate through relevant DataFrames
for idx in range(len(keys)): 
    # Subset to the relevant DataFrame
    key = keys[idx]
    combined_hof_df = hof_dataframes[key]
    print(f"Processing subdirectory: {key}")

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

    # Initialize df to hold results for each subdirectory
    results_df = pd.DataFrame(columns=["chem", "sympy_equation", "parital derivative w/ respect to chem", "integrated_derivative", "direction"])

    # Iterate through each chemical
    for j in range(len(chems)):  
        print(f'j_{j}')

        # Chemical of interest
        chem = chems[j]

        # Subset HOF to only equations containing this chemical
        subset_df = combined_hof_df[combined_hof_df['equation'].str.contains(rf'\b{chem}\b', na=False)]

        # Iterate through each row in the subset DataFrame
        for k in range(len(subset_df)):  
            print(f'k_{k}')

            try:
                # Get the equation 
                equation_str = subset_df.iloc[k]['equation']

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
                    ranges.append(chemical_ranges.get(sym_str)) 

                # Integrate the partial derivative over all variables 
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

    # Save results for each subdirectory
    results_file_name = f'Models/2_Chemical_measurements/pysr/partial_deriv_{key}.csv'
    results_df.to_csv(results_file_name, index=False)


# Calculate variable importance 
for idx in range(len(keys)):  
    # Subset to the relevant DataFrame
    key = keys[idx]

    # Get integrated info
    file_path = f'Models/2_Chemical_measurements/pysr/partial_deriv_{key}.csv'
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

        # Sum the 'integrated_derivative' for the current chemical
        sum_integrated_derivative = chem_rows['integrated_derivative'].sum()

        # Get chemical concentrations
        min_value, max_value = chemical_ranges[chem]
        range_value = max_value - min_value

        # Calculate the var_importance
        var_importance = sum_integrated_derivative * (len(chem_rows) / len(combined_hof_df)) / range_value

        # Log transform
        # var_importance = math.log(var_importance)

        # Append the chemical name and its var_importance to the list
        var_importance_list.append([chem, var_importance])

    # Create a new DataFrame for var_importance
    var_importance_df = pd.DataFrame(var_importance_list, columns=['chem', 'var_importance'])
    var_importance_df['chem'] = var_importance_df['chem'].astype(str)

    # Sort the DataFrame by 'var_importance' in decreasing order
    var_importance_df = var_importance_df.sort_values(by='var_importance', ascending=False)

    # Create a bar plot for each subdirectory
    plt.figure(figsize=(10, 6))
    plt.bar(var_importance_df['chem'], var_importance_df['var_importance'])
    plt.xlabel('Chemical')
    plt.ylabel('Log of Variable Importance')
    plt.title(f'Log of Variable Importance by Chemical ({key})')
    plt.xticks(rotation=90)  # Rotate chemical names for better readability
    plt.tight_layout()
    plt.show()

    # Save the plot for each subdirectory
    plt.savefig(f'Images/2_Chemical_measurements/pysr/var_importance_{key}.png')
