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
train_x = pd.read_pickle("Data_inputs/3_Omic_measurements/train_x")
test_x = pd.read_pickle("Data_inputs/3_Omic_measurements/test_x")
train_y = pd.read_pickle("Data_inputs/3_Omic_measurements/train_y")
test_y = pd.read_pickle("Data_inputs/3_Omic_measurements/test_y")

# Set up model    
# Model parameters
rf_model = RandomForestRegressor(n_estimators=10000, random_state=17)

# Run and time model
start_time = time.time()
rf_model.fit(train_x, train_y)
end_time = time.time()
time_taken = end_time - start_time

# Extract variable importance
importances = rf_model.feature_importances_
var_imp_rf_injury = pd.DataFrame({"Feature": train_x.columns, "Importance": importances})
var_imp_rf_injury = var_imp_rf_injury.sort_values("Importance", ascending=False)

# Save model comparisons to csv
file_name = f'Models/3_Omic_measurements/rf/rf_var_importance.csv'
var_imp_rf_injury.to_csv(file_name, index=False)

# Get top 15
var_imp_rf_injury = var_imp_rf_injury.iloc[:15,]

# Plot variable importance
plt.figure(figsize=(10, 6))
var_imp_rf_injury.plot(kind="bar", x="Feature", y="Importance")

# Adjust the layout
plt.tight_layout()

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

plt.title("Variable Importance Plot")
plt.xlabel("Feature")
plt.ylabel("Importance")

# Save the figure
plt.savefig(f'images/3_Omic_measurements/rf/var_importance.png')

# Get training data RMSE 
train_pred = rf_model.predict(train_x)
train_rmse = root_mean_squared_error(train_y, train_pred)

# Apply model to test set
test_pred = rf_model.predict(test_x)
test_rmse = root_mean_squared_error(test_y, test_pred)

# Store results in DataFrame
results_rf_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])
results_rf_df.loc[0] = [train_rmse, test_rmse, time_taken]

# Print final results
results_rf_df 

# Save model comparisons to csv
file_name = f'Models/3_Omic_measurements/rf/rf_model_comparison.csv'
results_rf_df.to_csv(file_name, index=False)
