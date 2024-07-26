import pysr
import sympy
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import root_mean_squared_error
import re
from feyn.plots import plot_regression
from feyn.plots import plot_residuals
import matplotlib.pyplot as plt
import os

# Set seed
np.random.seed(17)

# Load in input dictionaries and response variables
with open('Data_inputs/2_Chemical_measurements/train_input_dict.pkl', 'rb') as f:
    train_input_dict = pickle.load(f)    
with open('Data_inputs/2_Chemical_measurements/test_input_dict.pkl', 'rb') as f:
    test_input_dict = pickle.load(f)    
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")


# Run SRA using pysr
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


# Function to extract column names from a lambda_format equation
def extract_variables(equation):
    # Define the pattern to match variable names, including underscores
    pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    
    # Find all matches of the pattern in the equation
    matches = re.findall(pattern, equation)
    
    # Filter out common keywords that are not variable names
    keywords = {'X', 'PySRFunction', 'X=>'}
    
    # Remove duplicates and filter keywords
    variables = list({match for match in matches if match not in keywords})
    
    return variables

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

# Clean names
train_clean = {key: clean_column_names(df) for key, df in train_input_dict.items()}
test_clean = {key: clean_column_names(df) for key, df in test_input_dict.items()}

# Initialize a DataFrame to store results
results_pysr_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Define keys for loop
keys = list(train_clean.keys())

# Iterate through relevant inputs
for i in range(len(train_clean)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_clean[key]
    df_test = test_clean[key]

    # Define parameters - https://astroautomata.com/PySR/api/
    default_pysr_params = dict(
        populations = 15, # default number
        population_size = 33, #default number
        model_selection = "best",
        parsimony = 0.0032 # Multiplicative factor for how much to punish complexity - default value
    )

    # Define loss function - note triple quotes in julia allows line breaks 
    loss_function = """
    function f(tree, dataset::Dataset{T,L}, options) where {T,L}
        ypred, completed = eval_tree_array(tree, dataset.X, options)
        if !completed
            return L(Inf)
        end
        y = dataset.y
        return sqrt( (1 / length(y)) * sum(i -> (ypred[i] - y[i])^2, eachindex(y)))
    end
    """
    # Set up model 
    discovered_model = pysr.PySRRegressor(
        niterations=1,
        # unary_operators=['exp','log'],
        binary_operators=["-", "+", "*", "/"],
        # binary_operators=["-", "+", "*", "/", "^"],
        loss_function=loss_function,
        **default_pysr_params,
        temp_equation_file=True, 
        warm_start= True
    )

    # - https://github.com/MilesCranmer/PySR/discussions/439#discussioncomment-7330778
    # Initialize a DataFrame to store results
    results_rmse = pd.DataFrame(columns=["Iteration", "RMSE", "Complexity", "Equation"])

    # Start timer 
    start_time = time.time()

    # Iterate through generations
    for j in range(500):

        # Fit model
        discovered_model.fit(df_train.values, 
                        train_y.values,
                        variable_names=df_train.columns.tolist()
                        ) 
        
        # Extract equations
        equations = discovered_model.equations_
    
        # Evaluate RMSE for each equation
        for k in range(len(equations)):
            # Filter to the kth equation
            cur_eq = equations.iloc[k]['lambda_format']

            # Note complexity
            comp = equations.iloc[k]['complexity']
            
            # Subset training data to only retain the variables in the kth equation 
            cur_eq_str = str(cur_eq)
            cols = extract_variables(cur_eq_str)
            df_subset = df_train[cols]

            # Predict the outcome
            y_pred = evaluate_equation(cur_eq_str, df_subset)

            # Calculate RMSE
            rmse = root_mean_squared_error(y_pred, train_y)

            # Update results
            results_rmse = results_rmse._append({
                "Iteration": j + 1,
                "RMSE": rmse,
                "Complexity": comp,
                "Equation": cur_eq_str
            }, ignore_index=True)

    # Plot results every 100 iterations
        # if (j + 1) % 100 == 0:
        #     plt.figure()
        #     for comp in results_rmse['Complexity'].unique():
        #         subset = results_rmse[results_rmse['Complexity'] == comp]
        #         plt.plot(subset['Iteration'], subset['RMSE'], marker='.', linestyle='', label=f'Complexity {comp}')

        #     plt.xlabel('Iteration')
        #     plt.ylabel('RMSE')
        #     plt.title(f'{key} - Iteration {j + 1}')
        #     plt.legend()
        #     plt.show()

    # Save the complete plot with all iterations
    results_rmse['RMSE'] = np.log(results_rmse['RMSE'])
    plt.figure()
    for comp in results_rmse['Complexity'].unique():
        subset = results_rmse[results_rmse['Complexity'] == comp]
        plt.plot(subset['Iteration'], subset['RMSE'], marker='.', linestyle='', label=f'Complexity {comp}')

    plt.xlabel('Iteration')
    plt.ylabel('log(RMSE)')
    plt.title(f'{key} - RMSE Sensitivity')
    plt.legend()
    plt.savefig(f'images/2_Chemical_measurements/pysr/pysr_rmse_sensitivity_{key}_complete.png')

    # End timer and calculate time taken 
    end_time = time.time()
    time_taken = end_time - start_time

    # Save RMSE results to csv
    file_name = f'Models/2_Chemical_measurements/pysr/pysr_RMSE_sensitivity_{key}.csv'
    results_rmse.to_csv(file_name, index=False)

    # Get top 10 models
    equations = discovered_model.equations_
    df_equations = pd.DataFrame(equations) # Convert the equations to a DataFrame

    # Save the DataFrame to a CSV file
    file_name = f'Models/2_Chemical_measurements/pysr/pysr_HOF_{keys[i]}.csv'
    df_equations.to_csv(file_name, index=False)
    
    # Save the model to a file
    file_name = f'Models/2_Chemical_measurements/pysr/pysr_model_{keys[i]}.pkl'
    with open(file_name, 'wb') as model_file:
        pickle.dump(discovered_model, model_file)

    # Pysr Train RMSE 
    y_train = discovered_model.predict(df_train.values)
    train_pysr_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_pysr_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/2_Chemical_measurements/pysr/pysr_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/2_Chemical_measurements/pysr/pysr_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # Pysr Test RMSE 
    ypredict = discovered_model.predict(df_test.values)
    test_pysr_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_pysr_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/2_Chemical_measurements/pysr/pysr_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/2_Chemical_measurements/pysr/pysr_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_pysr_df.loc[key] = [train_pysr_rmse, test_pysr_rmse, time_taken]
    
# Print final results  
results_pysr_df

# Save model comparisons to csv
file_name = f'Models/2_Chemical_measurements/pysr/pysr_model_comparison.csv'
results_pysr_df.to_csv(file_name, index=False)