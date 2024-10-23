import feyn
from feyn.filters import ExcludeFunctions
from feyn.filters import ContainsFunctions
from feyn.tools import sympify_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import root_mean_squared_error
import time


# Load in data
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)    

# Define potential operator inputs
ops = [
    ["log:1", "exp:1", "sqrt:1", "squared:1", "inverse:1", "linear:1", "tanh:1", "gaussian:1", "gaussian:2", "multiply:2"],
    ["log:1", "exp:1", "sqrt:1", "squared:1", "inverse:1", "linear:1", "tanh:1", "gaussian:1", "gaussian:2"],
    ["squared:1", "inverse:1", "linear:1", "tanh:1", "gaussian:1", "gaussian:2"]
]

operators= pd.DataFrame({
    'operators': ops,
})
operators.index = ['low', 'med', 'high']

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Input", "Operators", "RMSE", "Equation"])

# Intialize row counting variable for results table
row = 0

# Start timer 
start_time = time.time()

# Iterate through operator options
for i in range(len(operators.index)):

    # Subset to relevant operators
    op_temp = operators.iloc[i, operators.columns.get_loc('operators')]

    # Iterate through different simulated inputs
    for j in range(len(sim_dict)):

        # Subset to one df
        keys = list(sim_dict.keys())
        key = keys[j]
        data = sim_dict[key]
        print(key)
        
        # Define input and output 
        x = data.drop('Response', axis = 1)
        y = data['Response']

        # Instantiate a QLattice
        ql = feyn.QLattice(random_seed=17)

        # Initate an empty list of models
        models = []

        # Compute prior probability of inputs based on mutual information
        priors = feyn.tools.estimate_priors(data, 'Response')

        # Update the QLattice with priors
        ql.update_priors(priors)

        for epoch in range(500): #https://docs.abzu.ai/docs/guides/primitives/using_primitives
            print(epoch)
            # Sample models from the QLattice, and add them to the list
            models += ql.sample_models(data, 'Response', 'regression', max_complexity=10)
            print(len(models))

            # Filter models
            f = ExcludeFunctions(ops[i])
            models = list(filter(f, models))
            print(len(models))
            
            # Remove redundant and poorly performing models from the list
            models = feyn.prune_models(models)
            print(len(models))

            # Fit the list of models. Returns a list of models sorted by absolute error 
            models = feyn.fit_models(models, data, 'absolute_error')

            # Update QLattice with the fitted list of models (sorted by loss)
            ql.update(models)

        # Calc RMSE
        y_pred = models[0].predict(x)
        rmse = root_mean_squared_error(y,y_pred)
        print(f"Training RMSE: {rmse:.2f}")

        # Save the DataFrame to a CSV file
        equations = []
        for k in range(10):
            eq = sympify_model(models[k])
            equations.append(eq)
        df_equations = pd.DataFrame(equations, columns=['Equation'])

        # Save to csv
        file_name = f'Models/1_Simulated_data/feyn/{list(operators.index)[i]}_{list(sim_dict.keys())[j]}_HOF.csv'
        df_equations.to_csv(file_name, index=False)

        # Save results
        results_df.loc[row]=[list(sim_dict.keys())[j], op_temp, rmse, equations[1]]
        row = row + 1
        
# End timer and calculate time taken 
end_time = time.time()
time_taken = end_time - start_time

results_df['time'] = time_taken

# Save to csv
results_df
file_name = f'Models/1_Simulated_data/feyn/results.csv'
results_df.to_csv(file_name, index=False)