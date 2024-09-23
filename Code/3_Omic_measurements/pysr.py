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

# Load in input dictionaries and response variables
train_x = pd.read_pickle("Data_inputs/3_Omic_measurements/train_x")
test_x = pd.read_pickle("Data_inputs/3_Omic_measurements/test_x")
train_y = pd.read_pickle("Data_inputs/3_Omic_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/3_Omic_measurements/test_y")

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

# Clean names for pysr
df_train = clean_column_names(train_x)
df_test = clean_column_names(test_x)

# Initialize a DataFrame to store results
results_pysr_df = pd.DataFrame(columns=["Input", "Training RMSE", "Test RMSE", "Time Taken"])

# Define parameters
default_pysr_params = dict(
        populations=15,  # default number
        population_size=33,  # default number
        model_selection="best",
        parsimony=0.0032  # Multiplicative factor for how much to punish complexity - default value
    )

# Define loss function
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

# Initialize model with warm_start set to True for continuing training
discovered_model = pysr.PySRRegressor(
        niterations=1,  # We will manually iterate
        binary_operators=["-", "+", "*", "/", "^"],
        loss_function=loss_function,
        **default_pysr_params,
        temp_equation_file=True,
        warm_start=True,  # Continue training from where the last iteration left off
        random_state=17
    )

# Start timer 
start_time = time.time()

# Iterate through a set number of iterations
for iteration in range(100):
    print(f"Iteration {iteration + 1}")

    # Fit the model (each iteration performs one step of fitting)
    discovered_model.fit(df_train.values, 
                             train_y.values,
                             variable_names=df_train.columns.tolist())
    # Get the Hall of Fame (HOF) after the iteration
    equations = discovered_model.equations_
    df_equations = pd.DataFrame(equations)  # Convert equations to a DataFrame

    # Save the HOF file for this iteration
    file_name = f'Models/3_Omic_measurements/HOF_all_iterations/hall_of_fame_iteration_{iteration+1}.csv'
    df_equations.to_csv(file_name, index=False)

# End timer and calculate time taken 
end_time = time.time()
time_taken = end_time - start_time

# Pysr Train RMSE 
y_train_predict = discovered_model.predict(df_train.values)
train_pysr_rmse = root_mean_squared_error(train_y, y_train_predict)
print(f"Training RMSE: {train_pysr_rmse:.2f}")

# Plot training residuals
file_name = f'images/3_Omic_measurements/pysr_residuals_train'
plot_residuals(train_y, y_train_predict, filename=file_name)

# Plot training regression plot
file_name = f'images/3_Omic_measurements/pysr_regression_train'
plot_regression(train_y, y_train_predict, filename=file_name)

# Pysr Test RMSE 
y_test_predict = discovered_model.predict(df_test.values)
test_pysr_rmse = root_mean_squared_error(test_y, y_test_predict)
print(f"Testing RMSE: {test_pysr_rmse:.2f}")

# Plot testing residuals
file_name = f'images/3_Omic_measurements/pysr_residuals_test'
plot_residuals(test_y, y_test_predict, filename=file_name)

# Plot testing regression plot
file_name = f'images/3_Omic_measurements/pysr_regression_test'
plot_regression(test_y, y_test_predict, filename=file_name)

# Store results in DataFrame
# results_pysr_df = [train_pysr_rmse, test_pysr_rmse, time_taken]

# Save model comparisons to csv after all iterations
# file_name = f'Models/3_Omic_measurements/pysr_model_comparison.csv'
# results_pysr_df.to_csv(file_name, index=False)

# Save predictions
file_name = f'Data_inputs/3_Omic_measurements/training_predictions'
pd.DataFrame(y_train_predict).to_pickle(file_name)
file_name = f'Data_inputs/3_Omic_measurements/test_predictions'
pd.DataFrame(y_test_predict).to_pickle(file_name)

# Print final results
print(results_pysr_df)