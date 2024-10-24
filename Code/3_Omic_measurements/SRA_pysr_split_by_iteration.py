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

# Set seed
np.random.seed(17)

# Load in input dictionaries and response variables
train_x = pd.read_pickle("Data_inputs/3_Omic_measurements/train_x")
test_x = pd.read_pickle("Data_inputs/3_Omic_measurements/test_x")
train_y = pd.read_pickle("Data_inputs/3_Omic_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/3_Omic_measurements/test_y")


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
    
    # Ensure all columns are numeric (convert non-numeric to NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Evaluate the formula using the subset DataFrame
    try:
        y_pred = df.eval(formula)
    except Exception as e:
        raise ValueError(f"Error evaluating the formula: {e}")
    
    # Create exception for when only a constant returned
    if isinstance(y_pred, (int, float)):  # Check if y_pred is a single number
        y_pred = pd.Series([y_pred] * len(df), index=df.index)
    
    return y_pred


# Clean names for pysr
df_train = clean_column_names(train_x)
df_test = clean_column_names(test_x)

# Initialize a DataFrame to store results
results_pysr_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE"])


# Define parameters - https://astroautomata.com/api/
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
        binary_operators=["-", "+", "*", "/", "^"],
        loss_function=loss_function,
        **default_pysr_params,
        temp_equation_file=True,
        warm_start=True,  # Continue training from where the last iteration left off
        random_state=17, 
        deterministic=True, 
        procs=0, 
        constraints={'^': (1, 1), 
                     '/': (-1, 2)},
        complexity_of_variables=2
    )


# - https://github.com/MilesCranmer/discussions/439#discussioncomment-7330778
# Initialize a DataFrame to store results
results_rmse = pd.DataFrame(columns=["Iteration", "RMSE", "Complexity", "Equation"])

# Iterate through generations
for j in range(1500):

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

# Save the complete plot with all iterations
results_rmse['RMSE'] = np.log(results_rmse['RMSE'])
plt.figure()
for comp in results_rmse['Complexity'].unique():
        subset = results_rmse[results_rmse['Complexity'] == comp]
        plt.plot(subset['Iteration'], subset['RMSE'], marker='.', linestyle='', label=f'Complexity {comp}')

plt.xlabel('Iteration')
plt.ylabel('log(RMSE)')
plt.title(f'RMSE Sensitivity')
plt.legend()
plt.savefig(f'images/3_Omic_measurements/pysr_rmse_sensitivity_complete.png')

# Save RMSE results to csv
file_name = f'Models/3_Omic_measurements/pysr_RMSE_sensitivity.csv'
results_rmse.to_csv(file_name, index=False)

# Get top 10 models
equations = discovered_model.equations_
df_equations = pd.DataFrame(equations) # Convert the equations to a DataFrame

# Save the DataFrame to a CSV file
file_name = f'Models/3_Omic_measurements/pysr_HOF.csv'
df_equations.to_csv(file_name, index=False)
    
# Pysr Train RMSE 
y_train = discovered_model.predict(df_train.values)
train_pysr_rmse = root_mean_squared_error(train_y, y_train)
print(f"Training RMSE: {train_pysr_rmse:.2f}")

# Plot training residuals
file_name = f'images/3_Omic_measurements/pysr_residuals_train'
plot_residuals(train_y, y_train, filename=file_name)

# Plot training regression plot
file_name = f'images/3_Omic_measurements/pysr_regression_train'
plot_regression(train_y, y_train, filename=file_name)

# Pysr Test RMSE 
ypredict = discovered_model.predict(df_test.values)
test_pysr_rmse = root_mean_squared_error(test_y, ypredict)
print(f"Testing RMSE: {test_pysr_rmse:.2f}")

# Plot testing residuals
file_name = f'images/3_Omic_measurements/pysr_residuals_test'
plot_residuals(test_y, ypredict, filename=file_name)

# Plot testing regression plot
file_name = f'images/3_Omic_measurements/pysr_regression_test'
plot_regression(test_y, ypredict, filename=file_name)

# Store results in DataFrame
results_pysr_df.loc = [train_pysr_rmse, test_pysr_rmse]
    
# Print final results  
results_pysr_df

# Save model comparisons to csv
file_name = f'Models/3_Omic_measurements/pysr_model_comparison.csv'
results_pysr_df.to_csv(file_name, index=False)