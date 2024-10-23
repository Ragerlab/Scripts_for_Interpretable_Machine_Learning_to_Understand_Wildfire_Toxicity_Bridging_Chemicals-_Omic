import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import ElasticNetCV

# Set seed
np.random.seed(17)

# Read in data
train_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x")
train_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_x = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x")
test_y = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_y")

# Reduce dimensionality using elastic
# Perform Elastic Net with cross-validation to select the best alpha (lambda) and l1_ratio
elastic_net_cv = ElasticNetCV(cv=3, l1_ratio=0.5, max_iter=1000).fit(train_x, train_y)

# Best alpha (lambda) value and l1_ratio value
best_alpha = elastic_net_cv.alpha_
best_l1_ratio = elastic_net_cv.l1_ratio_
print(f"Best alpha (lambda) value: {best_alpha}")
print(f"Best l1_ratio value: {best_l1_ratio}")

# Predicting using the best model
train_y_pred = elastic_net_cv.predict(train_x)

# Calculate RMSE for training data
rmse_train = np.sqrt(root_mean_squared_error(train_y, train_y_pred))
print(f"RMSE for the selected alpha on training data: {rmse_train}")

# Predict on test data
test_y_pred = elastic_net_cv.predict(test_x)
rmse_test = np.sqrt(root_mean_squared_error(test_y, test_y_pred))
print(f"RMSE for the selected alpha on test data: {rmse_test}")

# Get chemical coefficients
results_df = pd.DataFrame({
    'Variable': train_x.columns,
    'Coefficient': elastic_net_cv.coef_
})

# Subset to nonzero coefficients
nonzero_results = results_df[results_df['Coefficient'] != 0]
nonzero_results = nonzero_results.sort_values(by='Coefficient', ascending=False)
train_x_elastic = train_x[nonzero_results['Variable']]
test_x_elastic = test_x[nonzero_results['Variable']]

# Save reduced dataframes
nonzero_results.to_csv("Models/2_Chemical_measurements/Elastic/nonzero_coefs.csv")
train_x_elastic.to_pickle("Data_inputs/2_Chemical_measurements/train_x_elastic")
test_x_elastic.to_pickle("Data_inputs/2_Chemical_measurements/test_x_elastic")

