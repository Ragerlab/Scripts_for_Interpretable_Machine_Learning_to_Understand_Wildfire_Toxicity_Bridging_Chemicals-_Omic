import feyn
from feyn.filters import ExcludeFunctions
from feyn.filters import ContainsFunctions
from feyn.tools import sympify_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import root_mean_squared_error

# Load in data
with open('Data_inputs/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)    

# Define potential operator inputs
ops = [['add'], ['add', 'multiply'], ['add', 'multiply', 'sqrt', 'log']]
operators= pd.DataFrame({
    'operators': ops,
})
operators.index = ['low', 'med', 'high']

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Input", "Operators", "RMSE", "Equation"])

# Intialize row counting variable for results table
row = 0

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
        
        # Define input and output 
        x = data.drop('Response', axis = 1)
        y = data['Response']

        # Note column names to later fix gplearn output
        column_names = x.columns.tolist()
        column_mapping = {f'X{i}': name for i, name in enumerate(column_names)}

        # Instantiate a QLattice
        ql = feyn.QLattice()

        # Initate an empty list of models
        models = []

        # Compute prior probability of inputs based on mutual information
        priors = feyn.tools.estimate_priors(data, 'Response')

        # Update the QLattice with priors
        ql.update_priors(priors)

        for epoch in range(100): #https://docs.abzu.ai/docs/guides/primitives/using_primitives
            # Sample models from the QLattice, and add them to the list
            models += ql.sample_models(data, 'Response', 'regression')

            # Filter to models that only contain: +, =, *, /
            f = ContainsFunctions(op_temp)
            models = list(filter(f, models))

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

        # Compare equation predictions to actual
        pred = models[0].predict(x)
        plt.scatter(data['Response'], pred)
        plt.xlabel('Equation Output')
        plt.ylabel('Model Output')
        plt.title(file_name)
        # plt.show()

        # Save results
        results_df.loc[row]=[list(sim_dict.keys())[j], op_temp, rmse, equations[1]]
        row = row + 1

# Save to csv
results_df
file_name = f'Models/1_Simulated_data/feyn/results.csv'
results_df.to_csv(file_name, index=False)