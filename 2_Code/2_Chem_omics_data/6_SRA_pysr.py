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
import os

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Define dataset paths
datasets = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_y": "3_Data_intermediates/2_Chemical_measurements/Chem_train_y",
        "test_y": "3_Data_intermediates/2_Chemical_measurements/Chem_test_y",
        "train_input_dict": "3_Data_intermediates/2_Chemical_measurements/train_input_dict.pkl",
        "test_input_dict": "3_Data_intermediates/2_Chemical_measurements/test_input_dict.pkl",
        "iterations": 500
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_y": "3_Data_intermediates/3_Omic_measurements/Omic_train_y",
        "test_y": "3_Data_intermediates/3_Omic_measurements/Omic_test_y",
        "train_input_dict": "3_Data_intermediates/3_Omic_measurements/train_input_dict.pkl",
        "test_input_dict": "3_Data_intermediates/3_Omic_measurements/test_input_dict.pkl",
        "iterations": 1500
    }
]

# Function to clean column names for PySR
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

# Loop through datasets
for i in range(len(datasets)):
    dataset = datasets[i]
    print(f"Processing {dataset['prefix']} dataset...")

    # Load data
    train_y = pd.read_pickle(f"{dataset['train_y']}")
    test_y = pd.read_pickle(f"{dataset['test_y']}")
    with open(dataset["train_input_dict"], 'rb') as f:
        train_input_dict = pickle.load(f)
    with open(dataset["test_input_dict"], 'rb') as f:
        test_input_dict = pickle.load(f)

    # Clean names for PySR
    train_clean = {key: clean_column_names(df) for key, df in train_input_dict.items()}
    test_clean = {key: clean_column_names(df) for key, df in test_input_dict.items()}

    # Save cleaned names for later use
    output_data_path = f'3_Data_intermediates/{dataset["path"]}'
    os.makedirs(output_data_path, exist_ok=True)
    with open(f'{output_data_path}/train_clean.pkl', 'wb') as f:
        pickle.dump(train_clean, f)

    # Initialize a DataFrame to store results
    results_pysr_df = pd.DataFrame(columns=["Input", "Training RMSE", "Test RMSE", "Time Taken"])

    # Loop through train_clean inputs
    keys = list(train_clean.keys())
    for j in range(len(keys)):
        key = keys[j]
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

        # Run specified iterations
        for k in range(dataset["iterations"]):
            print(f"Iteration {k + 1} for {key}")

            # Fit model
            discovered_model.fit(
                df_train.values,
                train_y.values,
                variable_names=df_train.columns.tolist()
            )

            # Save Hall of Fame for each iteration
            equations = discovered_model.equations_
            df_equations = pd.DataFrame(equations)
            hof_dir = f'4_Model_results/{dataset["path"]}/pysr/HOF_all_iterations/{key}'
            os.makedirs(hof_dir, exist_ok=True)
            df_equations.to_csv(f'{hof_dir}/hall_of_fame_iteration_{k + 1}.csv', index=False)

        # End timer
        end_time = time.time()
        time_taken = end_time - start_time

        # Evaluate model
        y_train_predict = discovered_model.predict(df_train.values)
        train_rmse = root_mean_squared_error(train_y, y_train_predict)

        y_test_predict = discovered_model.predict(df_test.values)
        test_rmse = root_mean_squared_error(test_y, y_test_predict)

        # Save results
        results_pysr_df.loc[j] = [key, train_rmse, test_rmse, time_taken]

        # Save predictions
        pd.DataFrame(y_train_predict).to_pickle(f'{output_data_path}/{dataset["prefix"]}_training_predictions_pysr_{key}')
        pd.DataFrame(y_test_predict).to_pickle(f'{output_data_path}/{dataset["prefix"]}_test_predictions_pysr_{key}')

    # Save final results
    results_path = f'4_Model_results/{dataset["path"]}/pysr'
    os.makedirs(results_path, exist_ok=True)
    results_pysr_df.to_csv(f'{results_path}/pysr_model_comparison.csv', index=False)

