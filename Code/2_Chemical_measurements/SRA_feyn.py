import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import root_mean_squared_error
from feyn.plots import plot_regression
from feyn.plots import plot_residuals
import feyn
from feyn.filters import ExcludeFunctions
from feyn.filters import ContainsFunctions

# Load in input dictionaries and response variables
with open('Data_inputs/2_Chemical_measurements/train_input_dict.pkl', 'rb') as f:
    train_input_dict = pickle.load(f)    
with open('Data_inputs/2_Chemical_measurements/test_input_dict.pkl', 'rb') as f:
    test_input_dict = pickle.load(f)    
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")

# Initialize a DataFrame to store results
results_feyn_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Define keys for loop
keys = list(train_input_dict.keys())

# Iterate through relevant inputs
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_train['Injury_Protein'] = train_y
    df_test = test_input_dict[key]

    # Start timer
    start_time = time.time()

    # Instantiate a QLattice
    ql = feyn.QLattice()

    # Initate an empty list of models
    models = []

    # Compute prior probability of inputs based on mutual information
    priors = feyn.tools.estimate_priors(df_train, 'Injury_Protein')

    # Update the QLattice with priors
    ql.update_priors(priors)

    for epoch in range(1000):
        # Sample models from the QLattice, and add them to the list
        models += ql.sample_models(df_train, 'Injury_Protein', 'regression')

        # Filter to models that only contain: +, =, *, /
        # f = ContainsFunctions(["add", "multiply", "inverse"])
        # models = list(filter(f, models))

        # Fit the list of models. Returns a list of models sorted by absolute error 
        models = feyn.fit_models(models, df_train, 'absolute_error')

        # Display the best model in the current epoch
        feyn.show_model(models[0], label=f"Epoch: {epoch}", update_display=True)

        # Update QLattice with the fitted list of models (sorted by loss)
        ql.update(models)

    # Stop timer 
    end_time = time.time()
    time_taken = end_time - start_time

    # Run and time model 
    # start_time = time.time()
    # models = ql.auto_run(data=df_train,
    #                         output_name='Injury_Protein',
    #                         kind = 'regression',
    #                         n_epochs = 10,
    #                         loss_function = 'absolute_error'
    #                         )
    # end_time = time.time()
    # time_taken = end_time - start_time

    # Pull out the top 10 models
    top_10_models = models[:10]

    # Convert models to sympy readable format and store in a list
    sympy_models = []
    for model in top_10_models:
        sympy_model = model.sympify(signif=3)
        sympy_models.append(str(sympy_model.as_expr()))
    
    # Save the equations to a DataFrame
    df_equations = pd.DataFrame(sympy_models, columns=['Equation'])
    file_name = f'Models/feyn/feyn_HOF_{key}.csv'
    df_equations.to_csv(file_name, index=False)

    # Save the model to a file
    best = models[0]
    file_name = f'Models/2_Chemical_measurements/feyn/feyn_model_{keys[i]}.pkl'
    with open(file_name, 'wb') as model_file:
        pickle.dump(best, model_file)

    # gplearn Train RMSE 
    y_train = best.predict(df_train)
    train_gp_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_gp_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/2_Chemical_measurements/feyn/feyn_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/2_Chemical_measurements/feyn/feyn_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # gplearn Test RMSE 
    ypredict = best.predict(df_test)
    test_gp_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_gp_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/2_Chemical_measurements/feyn/feyn_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/2_Chemical_measurements/feyn/feyn_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_feyn_df.loc[key] = [train_gp_rmse, test_gp_rmse, time_taken]
results_feyn_df    

# Save model comparisons to csv
file_name = f'Models/2_Chemical_measurements/feyn/feyn_model_comparison.csv'
results_feyn_df.to_csv(file_name, index=False)