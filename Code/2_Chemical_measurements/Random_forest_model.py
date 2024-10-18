from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import root_mean_squared_error
import os

# Set seed
np.random.seed(17)

# Read in all inputs
# Full input
train_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x")
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")

# PCA
train_x_pca = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x_pca")
test_x_pca = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x_pca")

# Elastic
train_x_elastic = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x_elastic")
test_x_elastic= pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x_elastic")

# Create dictionaries containing full input, PCA-reduced input, and elastic-reduced input to iterate through 
train_input_dict = {'Full': train_x, 'PCA': train_x_pca, 'Elastic': train_x_elastic}
test_input_dict = {'Full': test_x, 'PCA': test_x_pca, 'Elastic': test_x_elastic}

# Save dictionaries for future use
with open('Data_inputs/2_Chemical_measurements/train_input_dict.pkl', 'wb') as f:
    pickle.dump(train_input_dict, f)
with open('Data_inputs/2_Chemical_measurements/test_input_dict.pkl', 'wb') as f:
    pickle.dump(test_input_dict, f)

# Convert dictionary keys to a list
keys = list(train_input_dict.keys())

# Initialize a DataFrame to store results
results_rf_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Iterate through inputs and run random forest model
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_test = test_input_dict[key]

    # Model parameters
    rf_model = RandomForestRegressor(n_estimators=10000, random_state=17)

    # Run and time model
    start_time = time.time()
    rf_model.fit(df_train, train_y)
    end_time = time.time()
    time_taken = end_time - start_time

    # Extract variable importance
    importances = rf_model.feature_importances_
    var_imp_rf_injury = pd.DataFrame({"Feature": df_train.columns, "Importance": importances})
    var_imp_rf_injury = var_imp_rf_injury.sort_values("Importance", ascending=False)

    # Plot variable importance
    plt.figure(figsize=(10, 6))
    var_imp_rf_injury.plot(kind="bar", x="Feature", y="Importance")
    plt.title("Variable Importance Plot")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig(f'images/2_Chemical_measurements/rf/var_importance_{key}.png')


    # Subset to the top 15 rows if the DataFrame has more than 15 rows
    if len(var_imp_rf_injury) > 15:
        var_imp_rf_injury = var_imp_rf_injury.head(15)

    # Plot variable importance
    plt.figure(figsize=(10, 6))
    ax = var_imp_rf_injury.plot(kind="bar", x="Feature", y="Importance", legend=False)
    plt.title("Variable Importance Plot")
    plt.xlabel("Feature")
    plt.ylabel("Importance")

    # Rotate x-axis labels diagonally for readability
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'images/2_Chemical_measurements/rf/var_importance_{key}.png')


    # Get training data RMSE 
    train_pred = rf_model.predict(df_train)
    train_rmse = root_mean_squared_error(train_y, train_pred)

    # Apply model to test set
    test_pred = rf_model.predict(df_test)
    test_rmse = root_mean_squared_error(test_y, test_pred)

    # Store results in DataFrame
    results_rf_df .loc[key] = [train_rmse, test_rmse, time_taken]

# Print final results
results_rf_df 

# Save model comparisons to csv
file_name = f'Models/2_Chemical_measurements/rf/rf_model_comparison.csv'
results_rf_df.to_csv(file_name, index=False)