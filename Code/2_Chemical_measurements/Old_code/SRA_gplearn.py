from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import root_mean_squared_error
from feyn.plots import plot_regression
from feyn.plots import plot_residuals
import graphviz
import matplotlib.pyplot as plt

# Set seed
np.random.seed(17)

# Load in input dictionaries and response variables
with open('Data_inputs/2_Chemical_measurements/train_input_dict.pkl', 'rb') as f:
    train_input_dict = pickle.load(f)    
with open('Data_inputs/2_Chemical_measurements/test_input_dict.pkl', 'rb') as f:
    test_input_dict = pickle.load(f)    
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")

# Calculate the arity (complexity) based on the equation strings
def calc_arity(cur_eq):
    # Initialize arity
    arity = 0
    
    # Define the operators that increase the arity
    operators = ['add', 'div', 'mul', 'sub']
    
    # Count occurrences of each operator in the string
    for operator in operators:
        arity += 2 * cur_eq.count(operator)
    
    return arity

# Define keys for loop
keys = list(train_input_dict.keys())

# Initialize the DataFrame to store the overall results
results_gp_df = pd.DataFrame(columns=["Key", "Train RMSE", "Test RMSE", "Time Taken"])

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
                               generations=1,
                               p_crossover=0.9,
                               metric='rmse',
                               function_set=('add', 'sub', 'mul', 'div'),
                               warm_start=True,
                               parsimony_coefficient= 0.01, 
                               feature_names=df_train.columns.tolist())
    
    # Initialize a DataFrame to store results
    results_rmse = pd.DataFrame(columns=["Iteration", "RMSE", "Complexity", "Equation"])

    # Start timer 
    start_time = time.time()
    # mod = est_gp.fit(df_train, train_y)

    # Iterate through generations one at a time
    for j in range(1000):
        # Fit model
        mod = est_gp.fit(df_train, train_y)

        # Extract the top 10 equations from the last generation
        top_eqs = mod._programs[-1][:10]

        # Extract the top 10 equations from the last generation
        sorted_programs = sorted(est_gp._programs[-1], key=lambda program: program.fitness_, reverse=True)
        top_eqs = sorted_programs[:10]

        # Evaluate RMSE for each equation
        for k in range(len(top_eqs)):
            # Filter to the kth equation
            cur_eq = top_eqs[k]

            # Evaluate each of the top equations
            y_train_pred = cur_eq.execute(df_train.values)
            rmse = np.sqrt(np.mean((train_y - y_train_pred) ** 2))

            # Calculate arity 
            arity = calc_arity(str(cur_eq))

            # Update results
            results_rmse = results_rmse._append({
                    "Iteration": est_gp.generations,
                    "RMSE": rmse,
                    "Complexity": k,
                    "Equation": str(cur_eq),
                }, ignore_index=True)
            
        # Increase generation
        est_gp.generations = j+2
            
    # End timer and calculate time taken 
    end_time = time.time()
    time_taken = end_time - start_time

    # Save the complete plot with all iterations
    # results_rmse['RMSE'] = np.log(results_rmse['RMSE'])
    # plt.figure()
    # for comp in results_rmse['Complexity'].unique():
    #     subset = results_rmse[results_rmse['Complexity'] == comp]
    #     plt.plot(subset['Iteration'], subset['RMSE'], marker='.', linestyle='', label=f'Complexity {comp}')

    # plt.xlabel('Iteration')
    # plt.ylabel('log(RMSE)')
    # plt.title(f'{key} - RMSE Sensitivity')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'images/2_Chemical_measurements/gplearn/gplearn_rmse_sensitivity_{key}_complete.png')
                                
    # Save RMSE results to csv
    file_name = f'Models/2_Chemical_measurements/gplearn/gplearn_RMSE_sensitivity_{key}.csv'
    results_rmse.to_csv(file_name, index=False)

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

    # Convert the equations to a DataFrame
    df_equations = pd.DataFrame(sympy_equations, columns=['Equation'])

    # Save the DataFrame to a CSV file
    file_name = f'Models/2_Chemical_measurements/gplearn/gplearn_HOF_{key}.csv'
    df_equations.to_csv(file_name, index=False)

    # Save the model to a file
    file_name = f'Models/2_Chemical_measurements/gplearn/gplearn_model_{keys[i]}.pkl'
    with open(file_name, 'wb') as model_file:
        pickle.dump(est_gp, model_file)

    # Plot final tree
    # dot_data = est_gp._program.export_graphviz()
    # graph = graphviz.Source(dot_data)
    # file_name = f'images/2_Chemical_measurements/gplearn/final_tree_{keys[i]}'
    # graph.render(file_name, format='png', cleanup=True)

    # gplearn Train RMSE 
    y_train =mod.predict(df_train.values)
    train_gp_rmse = np.sqrt(np.mean((train_y - y_train) ** 2))
    print(f"Training RMSE: {train_gp_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/2_Chemical_measurements/gplearn/gplearn_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/2_Chemical_measurements/gplearn/gplearn_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # gplearn Test RMSE 
    ypredict = mod.predict(df_test.values)
    test_gp_rmse = np.sqrt(np.mean((test_y - ypredict) ** 2))
    print(f"Testing RMSE: {test_gp_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/2_Chemical_measurements/gplearn/gplearn_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/2_Chemical_measurements/gplearn/gplearn_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_gp_df = results_gp_df._append({"Key": key, "Train RMSE": train_gp_rmse, "Test RMSE": test_gp_rmse, "Time Taken": time_taken}, ignore_index=True)

# Save model comparisons to csv
file_name = f'Models/2_Chemical_measurements/gplearn/gplearn_model_comparison.csv'
results_gp_df.to_csv(file_name, index=False)
