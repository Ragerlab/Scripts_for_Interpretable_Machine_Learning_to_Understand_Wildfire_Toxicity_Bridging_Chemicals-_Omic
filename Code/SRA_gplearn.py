from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import root_mean_squared_error
from feyn.plots import plot_regression
from feyn.plots import plot_residuals
import graphviz

# Load in input dictionaries and response variables
with open('Data_inputs/train_input_dict.pkl', 'rb') as f:
    train_input_dict = pickle.load(f)    
with open('Data_inputs/test_input_dict.pkl', 'rb') as f:
    test_input_dict = pickle.load(f)    
train_y = pd.read_pickle("Data_inputs/train_y")
test_y = pd.read_pickle("Data_inputs/test_y")

# Initialize a DataFrame to store results
results_gp_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Define keys for loop
keys = list(train_input_dict.keys())

# Iterate through relevant inputs
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_test = test_input_dict[key]

    # Note column names to later fix gplearn output
    column_names = df_train.columns.tolist()
    column_mapping = {f'X{i}': name for i, name in enumerate(column_names)}

    # Set model parameters - https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-regressor
    est_gp = SymbolicRegressor(population_size=1000,
                            generations=20, 
                            p_crossover=0.9, 
                            metric = 'rmse', 
                            function_set = ('add', 'sub', 'mul', 'div'))

    # Run and time model 
    start_time = time.time()
    est_gp.fit(df_train, train_y)
    end_time = time.time()
    time_taken = end_time - start_time

    # Print final equation 
    print(est_gp._program)

    # Extract the top 10 equations from the last generation
    top_10_equations = est_gp._programs[-1][:10]

    # Convert equations to sympy readable format and store in a list
    sympy_equations = []
    for program in top_10_equations:
            equation = str(program)
            for placeholder, actual_name in column_mapping.items():
                equation = equation.replace(placeholder, actual_name)
            sympy_equations.append(equation)

    sympy_equations = []
    for program in top_10_equations:
        sympy_equations.append(str(program))

    # Convert the equations to a DataFrame
    df_equations = pd.DataFrame(sympy_equations, columns=['Equation'])

    # Save the DataFrame to a CSV file
    file_name = f'Models/gplearn/gplearn_HOF_{key}.csv'
    df_equations.to_csv(file_name, index=False)

      # Save the model to a file
    file_name = f'Models/gplearn/gplearn_model_{keys[i]}.pkl'
    with open(file_name, 'wb') as model_file:
        pickle.dump(est_gp, model_file)

    # Plot final tree dot_data = est_gp._program.export_graphviz()
    # dot_data = est_gp._program.export_graphviz()
    # graph = graphviz.Source(dot_data)
    # file_name = f'images/gplearn/final_tree_{keys[i]}'
    # graph.render(file_name, format='png', cleanup=True)
    # graph

    # gplearn Train RMSE 
    y_train = est_gp.predict(df_train.values)
    train_gp_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_gp_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/gplearn/gplearn_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/gplearn/gplearn_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # gplearn Test RMSE 
    ypredict = est_gp.predict(df_test.values)
    test_gp_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_gp_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/gplearn/gplearn_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/gplearn/gplearn_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_gp_df.loc[key] = [train_gp_rmse, test_gp_rmse, time_taken]
results_gp_df