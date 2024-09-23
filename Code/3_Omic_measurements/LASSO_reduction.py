import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import root_mean_squared_error

# Set seed
np.random.seed(17)

# Read in data
train_x = pd.read_pickle("Data_inputs/3_Omic_measurements/train_x")
train_y = pd.read_pickle("Data_inputs/3_Omic_measurements/train_y")
test_x = pd.read_pickle("Data_inputs/3_Omic_measurements/test_x")

# Reduce dimensionality using Lasso
# Perform Lasso with cross-validation to select the best alpha (lambda)
lasso_cv = LassoCV(cv=3, max_iter = 1000).fit(train_x, train_y)

# Best alpha (lambda) value
best_alpha = lasso_cv.alpha_
print(f"Best alpha (lambda) value: {best_alpha}")

# Predicting using the best model
train_y_pred = lasso_cv.predict(train_x)

# Calculate RMSE
rmse = root_mean_squared_error(train_y, train_y_pred)
print(f"RMSE for the selected alpha: {rmse}")

# Get chemical coefficients
results_df = pd.DataFrame({
    'Variable': train_x.columns,
    'Coefficient': lasso_cv.coef_
})

# Subset to nonzero coefficients 
nonzero_results = results_df[results_df['Coefficient'] != 0]
train_x_lasso = train_x[nonzero_results['Variable']]
test_x_lasso = test_x[nonzero_results['Variable']]

# Save reduced dataframes
train_x_lasso.to_pickle("Data_inputs/3_Omic_measurements/train_x_lasso")
test_x_lasso.to_pickle("Data_inputs/3_Omic_measurements/test_x_lasso")

