import os
import pandas as pd
import pickle
import sympy as sp
import re
from scipy.integrate import quad
from scipy.integrate import nquad
import numpy as np

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

# Function to perform numerical integration over all variables
def integrate_over_all_variables(partial_derivative, all_symbols, ranges):
    try:
        # Convert the SymPy expression into a numerical function using lambdify
        func = sp.lambdify(all_symbols, partial_derivative, 'numpy')
        
        # Define nested integrals for numerical integration
        def nested_integral(index, *args):
            if index == len(all_symbols):
                # When all variables are integrated, return the evaluated function value
                return func(*args)
            else:
                var_range = ranges[index]
                lower_bound, upper_bound = var_range

                # Perform the integral over the current variable
                result, _ = quad(lambda x: nested_integral(index + 1, *args, x), lower_bound, upper_bound)
                return result

        # Start the recursive integration from the first variable (index 0)
        integrated_result = nested_integral(0)

        return integrated_result

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
# combined_hof_df = combined_hof_df.loc[:50,]

# Load in clean names 
with open('Data_inputs/2_Chemical_measurements/train_clean.pkl', 'rb') as f:
    train_clean = pickle.load(f)    
keys = list(train_clean.keys())[0]
chems = train_clean[keys].columns

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

    # Convert the chemical to a sympy symbol
    chem_symbol = sp.symbols(chem)

    # Iterate through each row in the subset DataFrame
    for j in range(len(subset_df)):
        print(j)
        try:
            # Get the equation string from the j-th row of the subset DataFrame
            equation_str = subset_df.iloc[j]['equation']

            # Convert the equation string to a sympy expression
            equation_sympy = sp.sympify(equation_str)

            # Compute the partial derivative of the equation with respect to the chemical
            partial_derivative = sp.diff(equation_sympy, chem_symbol)

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

        # Store the results for this row
        results_df = results_df._append({
            "chem": chem,
            "sympy_equation": equation_sympy if 'equation_sympy' in locals() else "Error in sympy parsing",
            "parital derivative w/ respect to chem": partial_derivative,
            "integrated_derivative": integrated_derivative,
            "direction": direction
        }, ignore_index=True)
        
# Save results
file_name = f'Models/2_Chemical_measurements/pysr/partial_deriv.csv'
results_df.to_csv(file_name, index=False)
