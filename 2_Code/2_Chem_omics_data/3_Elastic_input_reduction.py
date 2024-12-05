import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
import os

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Set seed
np.random.seed(17)

# Define paths for datasets
datasets = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_x": "3_Data_intermediates/2_Chemical_measurements/Chem_train_x",
        "train_y": "3_Data_intermediates/2_Chemical_measurements/Chem_train_y",
        "test_x": "3_Data_intermediates/2_Chemical_measurements/Chem_test_x",
        "test_y": "3_Data_intermediates/2_Chemical_measurements/Chem_test_y"
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_x": "3_Data_intermediates/3_Omic_measurements/Omic_train_x",
        "train_y": "3_Data_intermediates/3_Omic_measurements/Omic_train_y",
        "test_x": "3_Data_intermediates/3_Omic_measurements/Omic_test_x",
        "test_y": "3_Data_intermediates/3_Omic_measurements/Omic_test_y"
    }
]

# Loop through datasets using indices
for i in range(len(datasets)):
    dataset = datasets[i]
    print(f"Processing {dataset['prefix']} dataset...")

    # Load data
    train_x = pd.read_pickle(dataset["train_x"])
    train_y = pd.read_pickle(dataset["train_y"])
    test_x = pd.read_pickle(dataset["test_x"])
    test_y = pd.read_pickle(dataset["test_y"])

    # Apply Elastic Net with cross-validation
    elastic_net_cv = ElasticNetCV(cv=3, l1_ratio=0.5, max_iter=1000).fit(train_x, train_y)

    # Output the best alpha and l1_ratio
    best_alpha = elastic_net_cv.alpha_
    best_l1_ratio = elastic_net_cv.l1_ratio_
    print(f"{dataset['prefix']} - Best alpha (lambda): {best_alpha}")
    print(f"{dataset['prefix']} - Best l1_ratio: {best_l1_ratio}")

    # Predict and calculate RMSE
    train_y_pred = elastic_net_cv.predict(train_x)
    rmse_train = np.sqrt(mean_squared_error(train_y, train_y_pred))
    print(f"{dataset['prefix']} - RMSE (train): {rmse_train}")

    test_y_pred = elastic_net_cv.predict(test_x)
    rmse_test = np.sqrt(mean_squared_error(test_y, test_y_pred))
    print(f"{dataset['prefix']} - RMSE (test): {rmse_test}")

    # Extract nonzero coefficients
    results_df = pd.DataFrame({
        'Variable': train_x.columns,
        'Coefficient': elastic_net_cv.coef_
    })
    nonzero_results = results_df[results_df['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False)

    # Subset data to nonzero coefficients
    train_x_elastic = train_x[nonzero_results['Variable']]
    test_x_elastic = test_x[nonzero_results['Variable']]

    # Save results
    output_dir = f"4_Model_results/{dataset['path']}/Elastic"
    os.makedirs(output_dir, exist_ok=True)
    nonzero_results.to_csv(f"{output_dir}/{dataset['prefix']}_nonzero_coefs.csv", index=False)
    train_x_elastic.to_pickle(f"3_Data_intermediates/{dataset['path']}/{dataset['prefix']}_train_x_elastic")
    test_x_elastic.to_pickle(f"3_Data_intermediates/{dataset['path']}/{dataset['prefix']}_test_x_elastic")

    print(f"Finished processing {dataset['prefix']} dataset.\n")
