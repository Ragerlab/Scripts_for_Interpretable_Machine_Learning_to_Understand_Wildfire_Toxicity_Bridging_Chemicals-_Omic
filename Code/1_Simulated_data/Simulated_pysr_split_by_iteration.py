import pandas as pd
import numpy as np
import pysr
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import root_mean_squared_error
import re

# Load in data
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)    

# Function to extract column names from a lambda_format equation
def extract_variables(equation):
    # Define the pattern to match variable names, including underscores
    pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    
    # Find all matches of the pattern in the equation
    matches = re.findall(pattern, equation)
    
    # Filter out common keywords that are not variable names
    keywords = {'X', 'PySRFunction', 'X=>', 'sqrt', 'log'}
    
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

# Define potential operator inputs
un_vec = [["myfunction(x) = x"], ["myfunction(x) = x"], ['log', 'sqrt']]
bin_vec = [['-', '+'], ['-', '+', '*', '/'], ['-', '+', '*', '/']]
operators= pd.DataFrame({
    'unary': un_vec,
    'binary': bin_vec,
})
operators.index = ['low', 'med', 'high']

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Input", "Binary operators", "Unary Operators", "RMSE", "Equation"])

# Intialize row counting variable for results table
row = 0

# Iterate through operator options
for i in range(len(operators.index)):

    # Subset to relevant operators
    bin = operators.iloc[i, operators.columns.get_loc('binary')]
    un = operators.iloc[i, operators.columns.get_loc('unary')]

    # Iterate through different simulated inputs
    for j in range(len(sim_dict)):

        # Subset to one df
        keys = list(sim_dict.keys())
        key = keys[j]
        data = sim_dict[key]
        
        # Define input and output 
        x = data.drop('Response', axis = 1)
        y = data['Response']

        # Define parameters - https://astroautomata.com/PySR/api/
        default_pysr_params = dict(
                populations = 30, # default number
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
                unary_operators=un,
                binary_operators= bin,
                loss_function=loss_function,
                **default_pysr_params,
                temp_equation_file=True, 
                extra_sympy_mappings={'myfunction': lambda x: x},
                warm_start= True
            )
        
        # Initialize a DataFrame to store results
        results_rmse = pd.DataFrame(columns=["Iteration", "RMSE", "Complexity", "Equation"])

        # Iterate through generations
        for k in range(10):

            # Fit model
            discovered_model.fit(x.values, 
                            y.values,
                            variable_names=x.columns.tolist()
                            ) 
            
            # Extract equations
            equations = discovered_model.equations_
        
            # Evaluate RMSE for each equation
            for l in range(len(equations)):
                # Filter to the kth equation
                cur_eq = equations.iloc[l]['lambda_format']

                # Note complexity
                comp = equations.iloc[l]['complexity']
                
                # Subset training data to only retain the variables in the kth equation 
                cur_eq_str = str(cur_eq)
                cols = extract_variables(cur_eq_str)
                df_subset = x[cols]

                # Predict the outcome
                y_pred = evaluate_equation(cur_eq_str, df_subset)

                # Calculate RMSE
                rmse = root_mean_squared_error(y_pred, y)

                # Update results
                results_rmse = results_rmse._append({
                    "Iteration": k + 1,
                    "RMSE": rmse,
                    "Complexity": comp,
                    "Equation": cur_eq_str
                }, ignore_index=True)

        # Compare equation predictions to actual
        pred = discovered_model.predict(x)
        plt.figure()
        plt.scatter(data['Response'], pred)
        plt.xlabel('Equation Output')
        plt.ylabel('Model Output')
        plt.title(file_name)
        # plt.show()
        plt.savefig(f'images/1_Simulated_data/pysr/real_vs_pred_{key}.png')

    # Save the complete plot with all iterations
    results_rmse['RMSE'] = np.log(results_rmse['RMSE'])
    plt.figure()
    for comp in results_rmse['Complexity'].unique():
        subset = results_rmse[results_rmse['Complexity'] == comp]
        plt.plot(subset['Iteration'], subset['RMSE'], marker='.', linestyle='', label=f'Complexity {comp}')

    plt.xlabel('Iteration')
    plt.ylabel('log(RMSE)')
    plt.title(f'{key} - RMSE Sensitivity')
    # Place legend outside of the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    plt.savefig(f'images/1_Simulated_data/pysr/pysr_rmse_sensitivity_{key}_complete.png')

    # Save RMSE results to csv
    file_name = f'Models/1_Simulated_data/pysr/pysr_RMSE_sensitivity_{key}.csv'
    results_rmse.to_csv(file_name, index=False)

    # Get top model
    best_model = discovered_model.get_best()
    best_equation = best_model['equation']

    # Get top 10 models
    equations = pd.DataFrame(discovered_model.equations_)

    # Save the DataFrame to a CSV file
    file_name = f'Models/1_Simulated_data/pysr/{list(operators.index)[i]}_{list(sim_dict.keys())[j]}_HOF.csv'
    equations.to_csv(file_name, index=False)

    # Calc RMSE
    y_pred = discovered_model.predict(x)
    rmse = root_mean_squared_error(y,y_pred)
    print(f"Training RMSE: {rmse:.2f}")

    # Save results
    results_df.loc[row]=[list(sim_dict.keys())[j], bin, un, rmse, best_equation]
    row = row + 1

# Save to csv
results_df
file_name = f'Models/1_Simulated_data/pysr/results_pysr.csv'
results_df.to_csv(file_name, index=False)
