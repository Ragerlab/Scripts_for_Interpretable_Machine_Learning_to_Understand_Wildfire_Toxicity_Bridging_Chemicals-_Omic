import pysr
import sympy
import pandas as pd
import numpy as np
import pickle
import time
import os
import re
from sklearn.metrics import root_mean_squared_error
from feyn.plots import plot_regression, plot_residuals
import matplotlib.pyplot as plt

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Define dataset paths
datasets = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_y": "3_Data_intermediates/2_Chemical_measurements/Chem_train_y",
        "test_y": "3_Data_intermediates/2_Chemical_measurements/Chem_test_y",
        "train_input_dict": "3_Data_intermediates/2_Chemical_measurements/train_input_dict.pkl",
        "test_input_dict": "3_Data_intermediates/2_Chemical_measurements/test_input_dict.pkl",
        "iterations": 1000
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_y": "3_Data_intermediates/3_Omic_measurements/Omic_train_y",
        "test_y": "3_Data_intermediates/3_Omic_measurements/Omic_test_y",
        "train_input_dict": "3_Data_intermediates/3_Omic_measurements/train_input_dict.pkl",
        "test_input_dict": "3_Data_intermediates/3_Omic_measurements/test_input_dict.pkl",
        "iterations": 3000
    }
]

# Function to clean column names for PySR
def clean_column_names(df):
    new_columns = []
    for col in df.columns:
        cleaned_col = re.sub(r'\W+', '', col)
        cleaned_col = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', cleaned_col)
        cleaned_col = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', cleaned_col)
        if cleaned_col in ['S', 'Si']:
            cleaned_col += "_var"
        if cleaned_col[0].isdigit():
            cleaned_col = 'var' + cleaned_col
        new_columns.append(cleaned_col)
    df.columns = new_columns
    return df

# Update function to check and replace problematic names
def assert_valid_sympy_symbol(var_name):
    invalid_names = {'S', 'Si'}
    if var_name in invalid_names:
        return var_name + "_var"
    return var_name

# Update extract_variables function to apply name correction
def extract_variables(equation):
    pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    matches = re.findall(pattern, equation)
    keywords = {'X', 'PySRFunction', 'X=>'}
    return [assert_valid_sympy_symbol(match) for match in matches if match not in keywords]

# Function to evaluate lambda_format equation
def evaluate_equation(equation, df):
    # Remove the prefix 'PySRFunction(X=>'
    formula = equation.replace('PySRFunction(X=>', '').rstrip(')')
    
    # Check for and correct missing parentheses
    open_parentheses = formula.count('(')
    close_parentheses = formula.count(')')
    if open_parentheses > close_parentheses:
        formula += ')' * (open_parentheses - close_parentheses)
    
    # Evaluate the formula using the subset DataFrame
    y_pred = df.eval(formula)

    # Create exception for when only a constant returned
    if isinstance(y_pred, (int, float)):  # Check if y_pred is a single number
            y_pred = pd.Series([y_pred] * len(df), index=df.index)
    
    return y_pred

# Loop through datasets
for dataset in datasets:
    print(f"Processing {dataset['prefix']} dataset...")
    
    # Load data
    train_y = pd.read_pickle(dataset['train_y'])
    test_y = pd.read_pickle(dataset['test_y'])
    with open(dataset['train_input_dict'], 'rb') as f:
        train_input_dict = pickle.load(f)
    with open(dataset['test_input_dict'], 'rb') as f:
        test_input_dict = pickle.load(f)
    
    # Clean names for PySR
    train_clean = {key: clean_column_names(df) for key, df in train_input_dict.items()}
    test_clean = {key: clean_column_names(df) for key, df in test_input_dict.items()}
    
    output_data_path = f'3_Data_intermediates/{dataset["path"]}'
    os.makedirs(output_data_path, exist_ok=True)
    
    results_pysr_df = pd.DataFrame(columns=["Input", "Training RMSE", "Test RMSE", "Time Taken"])
    
    for key, df_train in train_clean.items():
        df_test = test_clean[key]
        
        default_pysr_params = dict(
            populations=15,
            population_size=33,
            model_selection="best",
            parsimony=0.0032
        )

        discovered_model = pysr.PySRRegressor(
            niterations=1,
            binary_operators=["-", "+", "*", "/", "^"],
            **default_pysr_params,
            temp_equation_file=True,
            warm_start=True,
            random_state=17,
            deterministic=True,
            procs=0,
            constraints={'^': (1, 1), '/': (-1, 2)},
            complexity_of_variables=2
        )
        
        results_rmse = pd.DataFrame(columns=["Iteration", "RMSE", "Complexity", "Equation"])
        for j in range(dataset['iterations']):
            discovered_model.fit(df_train.values, train_y.values, variable_names=df_train.columns.tolist())
            equations = discovered_model.equations_
                        
            for k in range(len(equations)):
                cur_eq = equations.iloc[k]['lambda_format']
                comp = equations.iloc[k]['complexity']
                cols = extract_variables(str(cur_eq))
                df_subset = df_train[cols]
                df_subset = df_subset.loc[:, ~df_subset.columns.duplicated()]
                y_pred = evaluate_equation(str(cur_eq), df_subset)
                rmse = root_mean_squared_error(y_pred, train_y)
                
                results_rmse = results_rmse._append({
                    "Iteration": j + 1,
                    "RMSE": rmse,
                    "Complexity": comp,
                    "Equation": str(cur_eq)
                }, ignore_index=True)
            
        results_rmse.to_csv(f'4_Model_results/{dataset["path"]}/pysr/pysr_RMSE_sensitivity_{key}.csv', index=False)
    

