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
with open('Data_inputs/2_Chemical_measurements/train_input_dict.pkl', 'rb') as f:
    train_input_dict = pickle.load(f)    
with open('Data_inputs/2_Chemical_measurements/test_input_dict.pkl', 'rb') as f:
    test_input_dict = pickle.load(f)    
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")

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
train_clean = {key: clean_column_names(df) for key, df in train_input_dict.items()}
test_clean = {key: clean_column_names(df) for key, df in test_input_dict.items()}

# Save a copy of clean names to access later 
with open('Data_inputs/2_Chemical_measurements/train_clean.pkl', 'wb') as f:
    pickle.dump(train_clean, f)

# Initialize a DataFrame to store results
results_pysr_df = pd.DataFrame(columns=["Input", "Training RMSE", "Test RMSE", "Time Taken"])

# Define keys for loop
keys = list(train_clean.keys())

# Iterate through relevant inputs
for i in range(len(train_clean)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_clean[key]
    df_test = test_clean[key]

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

    # Start timer 
    start_time = time.time()

    # Iterate through a set number of iterations
    for iteration in range(500):
        print(f"Iteration {iteration + 1}")

        # Fit the model (each iteration performs one step of fitting)
        discovered_model.fit(df_train.values, 
                             train_y.values,
                             variable_names=df_train.columns.tolist())

        # Get the Hall of Fame (HOF) after the iteration
        equations = discovered_model.equations_
        df_equations = pd.DataFrame(equations)  # Convert equations to a DataFrame

        # Save the HOF file for this iteration
        file_name = f'Models/2_Chemical_measurements/pysr/HOF_all_iterations/{keys[i]}/hall_of_fame_iteration_{iteration+1}.csv'
        df_equations.to_csv(file_name, index=False)

    # End timer and calculate time taken 
    end_time = time.time()
    time_taken = end_time - start_time

    # Pysr Train RMSE 
    y_train_predict = discovered_model.predict(df_train.values, 6)
    train_pysr_rmse = root_mean_squared_error(train_y, y_train_predict)
    print(f"Training RMSE: {train_pysr_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/2_Chemical_measurements/pysr/pysr_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train_predict, filename=file_name)

    # Plot training regression plot
    file_name = f'images/2_Chemical_measurements/pysr/pysr_regression_train_{keys[i]}'
    plot_regression(train_y, y_train_predict, filename=file_name)

    # Pysr Test RMSE 
    y_test_predict = discovered_model.predict(df_test.values, 6)
    test_pysr_rmse = root_mean_squared_error(test_y, y_test_predict)
    print(f"Testing RMSE: {test_pysr_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/2_Chemical_measurements/pysr/pysr_residuals_test_{keys[i]}'
    plot_residuals(test_y, y_test_predict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/2_Chemical_measurements/pysr/pysr_regression_test_{keys[i]}'
    plot_regression(test_y, y_test_predict, filename=file_name)

    # Store results in DataFrame
    results_pysr_df.loc[i] = [key, train_pysr_rmse, test_pysr_rmse, time_taken]

    # Save predictions
    file_name = f'Data_inputs/2_Chemical_measurements/training_predictions_{key}'
    pd.DataFrame(y_train_predict).to_pickle(file_name)
    file_name = f'Data_inputs/2_Chemical_measurements/test_predictions_{key}'
    pd.DataFrame(y_test_predict).to_pickle(file_name)

# Print final results
print(results_pysr_df)

# Save model comparisons to csv after all iterations
file_name = f'Models/2_Chemical_measurements/pysr/pysr_model_comparison.csv'
results_pysr_df.to_csv(file_name, index=False)
