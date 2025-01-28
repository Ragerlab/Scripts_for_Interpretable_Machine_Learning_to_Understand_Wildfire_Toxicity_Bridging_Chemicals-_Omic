from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import mean_squared_error
import os

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Set seed
np.random.seed(17)

# Define dataset paths
datasets = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_x": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x",
        "train_y": "3_Data_intermediates/2_Chemical_measurements/Chem_train_y",
        "test_x": "3_Data_intermediates/2_Chemical_measurements/Chem_test_x",
        "test_y": "3_Data_intermediates/2_Chemical_measurements/Chem_test_y",
        "train_x_pca": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x_pca",
        "test_x_pca": "3_Data_intermediates/2_Chemical_measurements/Chem_test_x_pca",
        "train_x_elastic": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x_elastic",
        "test_x_elastic": "3_Data_intermediates/2_Chemical_measurements/Chem_test_x_elastic"
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_x_deg": "3_Data_intermediates/3_Omic_measurements/Omic_train_x_deg", 
        "train_y": "3_Data_intermediates/3_Omic_measurements/Omic_train_y",
        "test_x_deg": "3_Data_intermediates/3_Omic_measurements/Omic_test_x_deg",
        "test_y": "3_Data_intermediates/3_Omic_measurements/Omic_test_y",
        "train_x_pca": "3_Data_intermediates/3_Omic_measurements/Omic_train_x_pca",
        "test_x_pca": "3_Data_intermediates/3_Omic_measurements/Omic_test_x_pca",
        "train_x_elastic": "3_Data_intermediates/3_Omic_measurements/Omic_train_x_elastic",
        "test_x_elastic": "3_Data_intermediates/3_Omic_measurements/Omic_test_x_elastic"
    }, 
    {   
        "prefix": "Combined",
        "path": "4_ChemOmics_measurements",
        "train_x": "3_Data_intermediates/4_ChemOmics_measurements/Comb_train_x",
        "train_y": "3_Data_intermediates/4_ChemOmics_measurements/Comb_train_y",
        "test_x": "3_Data_intermediates/4_ChemOmics_measurements/Comb_test_x",
        "test_y": "3_Data_intermediates/4_ChemOmics_measurements/Comb_test_y", 
        "train_x_pca": "3_Data_intermediates/4_ChemOmics_measurements/Comb_train_x_pca",
        "test_x_pca": "3_Data_intermediates/4_ChemOmics_measurements/Comb_test_x_pca",
        "train_x_elastic": "3_Data_intermediates/4_ChemOmics_measurements/Comb_train_x_elastic",
        "test_x_elastic": "3_Data_intermediates/4_ChemOmics_measurements/Comb_test_x_elastic"
    }
]

# Loop through datasets
for i in range(len(datasets)):
    dataset = datasets[i]
    print(f"Processing {dataset['prefix']} dataset...")

    # Load all inputs
    train_y = pd.read_pickle(dataset["train_y"])
    test_y = pd.read_pickle(dataset["test_y"])
    
    if dataset["prefix"] != "Omic":
        # For the Chem dataset
        train_input_dict = {
            'Full': pd.read_pickle(dataset["train_x"]),
            'PCA': pd.read_pickle(dataset["train_x_pca"]),
            'Elastic': pd.read_pickle(dataset["train_x_elastic"])
        }
        test_input_dict = {
            'Full': pd.read_pickle(dataset["test_x"]),
            'PCA': pd.read_pickle(dataset["test_x_pca"]),
            'Elastic': pd.read_pickle(dataset["test_x_elastic"])
        }
    elif dataset["prefix"] == "Omic":
        # For the Omic dataset
        train_input_dict = {
            'DEG': pd.read_pickle(dataset["train_x_deg"]),
            'PCA': pd.read_pickle(dataset["train_x_pca"]),
            'Elastic': pd.read_pickle(dataset["train_x_elastic"])
        }
        test_input_dict = {
            'DEG': pd.read_pickle(dataset["test_x_deg"]),
            'PCA': pd.read_pickle(dataset["test_x_pca"]),
            'Elastic': pd.read_pickle(dataset["test_x_elastic"])
        }

    # Save dictionaries for future use
    output_data_path = f'3_Data_intermediates/{dataset["path"]}'
    os.makedirs(output_data_path, exist_ok=True)
    with open(f'{output_data_path}/train_input_dict.pkl', 'wb') as f:
        pickle.dump(train_input_dict, f)
    with open(f'{output_data_path}/test_input_dict.pkl', 'wb') as f:
        pickle.dump(test_input_dict, f)

    # Initialize a DataFrame to store results
    results_rf_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

    # Iterate through inputs and run random forest model
    for key, train_x in train_input_dict.items():
        test_x = test_input_dict[key]
        
        # Model parameters
        rf_model = RandomForestRegressor(n_estimators=10000, random_state=17)

        # Run and time model
        start_time = time.time()
        rf_model.fit(train_x, train_y)
        end_time = time.time()
        time_taken = end_time - start_time

        # Extract variable importance
        importances = rf_model.feature_importances_
        var_imp_rf = pd.DataFrame({"Feature": train_x.columns, "Importance": importances})
        var_imp_rf = var_imp_rf.sort_values("Importance", ascending=False)
        
        # Save variable importance data
        var_imp_rf.to_csv(f'{output_data_path}/rf_variable_importance_{key}.csv', index=False)

        # Create directories for images
        output_img_path = f'5_Plots/{dataset["path"]}/rf'
        os.makedirs(output_img_path, exist_ok=True)

        # Save variable importance plot
        plt.figure(figsize=(10, 6))
        var_imp_rf.head(15).plot(kind="bar", x="Feature", y="Importance", legend=False)
        plt.title(f"Variable Importance Plot ({key})")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_img_path}/var_importance_{key}.png')
        plt.close()

        # Calculate RMSE
        train_pred = rf_model.predict(train_x)
        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))

        test_pred = rf_model.predict(test_x)
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))

        # Store results
        results_rf_df.loc[key] = [train_rmse, test_rmse, time_taken]

        # Save predictions
        pd.DataFrame(train_pred).to_pickle(f'{output_data_path}/training_predictions_rf_{key}')
        pd.DataFrame(test_pred).to_pickle(f'{output_data_path}/test_predictions_rf_{key}')

    # Save model comparisons to CSV
    output_model_path = f'4_Model_results/{dataset["path"]}/rf'
    os.makedirs(output_model_path, exist_ok=True)
    results_rf_df.to_csv(f'{output_model_path}/rf_model_comparison.csv', index=False)
