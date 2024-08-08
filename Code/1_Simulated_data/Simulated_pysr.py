import pandas as pd
import numpy as np
import pysr
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import root_mean_squared_error
import time

# Load in data
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)    


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

# Start timer 
start_time = time.time()

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
                niterations=500,
                unary_operators=un,
                binary_operators= bin,
                loss_function=loss_function,
                **default_pysr_params,
                temp_equation_file=True, 
                extra_sympy_mappings={'myfunction': lambda x: x},
                warm_start= True
            )

        # Run model 
        discovered_model.fit(x.values, 
                                y.values,
                                variable_names=x.columns.tolist()
                                )    
        print(discovered_model.fit)

        # Get top model
        best_model = discovered_model.get_best()
        best_equation = best_model['equation']

        # Get top 10 models
        equations = pd.DataFrame(discovered_model.equations_)

        # Save the DataFrame to a CSV file
        file_name = f'Models/1_Simulated_data/pysr/{list(operators.index)[i]}_{list(sim_dict.keys())[j]}_HOF.csv'
        equations.to_csv(file_name, index=False)

        # Compare equation predictions to actual
        pred = discovered_model.predict(x)
        plt.figure()
        plt.scatter(data['Response'], pred)
        plt.xlabel('Equation Output')
        plt.ylabel('Model Output')
        plt.title(file_name)
        # plt.show()
        plt.savefig(f'images/1_Simulated_data/pysr/real_vs_pred_{key}.png')

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

# End timer and calculate time taken 
end_time = time.time()
time_taken = end_time - start_time

results_df['time'] = time_taken

# Save to csv
results_df
file_name = f'Models/1_Simulated_data/pysr/results_pysr.csv'
results_df.to_csv(file_name, index=False)
