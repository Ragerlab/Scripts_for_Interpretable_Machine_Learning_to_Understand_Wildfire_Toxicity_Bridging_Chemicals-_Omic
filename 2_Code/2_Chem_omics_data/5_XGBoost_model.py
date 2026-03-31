from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import time
import os
import re

# XGBoost
from xgboost import XGBRegressor

# Set working directory to project root (two levels up from this script)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

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
    }
]

def to_1d(y):
    """Ensure y is a 1D array/Series for sklearn/xgboost."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            return y.iloc[:, 0]
        else:
            raise ValueError("y has multiple columns; this script assumes 1D regression target.")
    return y.squeeze()

def make_xgb_safe_columns(df):
    """Return df with sanitized columns + mappings."""
    orig2clean, clean2orig = {}, {}
    used, newcols = set(), []
    for c in df.columns:
        s = str(c)
        s = re.sub(r"[\[\]<>]", "", s)          # remove [ ] <
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r"\s+", "_", s)              # spaces -> _
        s = re.sub(r"[^0-9A-Za-z_.:-]", "_", s) # only safe chars
        s = s.strip("_")
        if s == "":
            s = "feat"
        base, k = s, 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        orig2clean[c] = s
        clean2orig[s] = c
        newcols.append(s)
    df_clean = df.copy()
    df_clean.columns = newcols
    return df_clean, orig2clean, clean2orig

# Loop through datasets
for dataset in datasets:
    print(f"Processing {dataset['prefix']} dataset...")

    # Load labels
    train_y = pd.read_pickle(dataset["train_y"])
    test_y = pd.read_pickle(dataset["test_y"])
    train_y = to_1d(train_y)
    test_y = to_1d(test_y)

    # Load inputs
    if dataset["prefix"] != "Omic":
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
    else:
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

    # Save dictionaries
    output_data_path = f'3_Data_intermediates/{dataset["path"]}'
    os.makedirs(output_data_path, exist_ok=True)
    with open(f'{output_data_path}/train_input_dict.pkl', 'wb') as f:
        pickle.dump(train_input_dict, f)
    with open(f'{output_data_path}/test_input_dict.pkl', 'wb') as f:
        pickle.dump(test_input_dict, f)

    # ========= XGBoost =========
    results_xgb_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken (s)"])

    for key, train_x in train_input_dict.items():
        test_x = test_input_dict[key]

        # Sanitize column names
        train_x_clean, o2c, c2o = make_xgb_safe_columns(train_x)
        test_x_clean = test_x.rename(columns=o2c).copy()

        # Align test columns
        missing_in_test = [c for c in train_x_clean.columns if c not in test_x_clean.columns]
        for c in missing_in_test:
            test_x_clean[c] = 0.0
        test_x_clean = test_x_clean[train_x_clean.columns]

        # Model
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=17,
            n_jobs=-1,
            tree_method="hist",
            importance_type="gain",
            eval_metric="rmse"
        )

        start_time = time.time()
        xgb_model.fit(train_x_clean, train_y)
        time_taken = time.time() - start_time

        # Variable importance (map back to original names)
        importances = xgb_model.feature_importances_
        var_imp_xgb = (
            pd.DataFrame({
                "Feature": [c2o[c] for c in train_x_clean.columns],
                "Importance": importances
            })
            .sort_values("Importance", ascending=False)
        )
        var_imp_xgb.to_csv(f'{output_data_path}/xgb_variable_importance_{key}.csv', index=False)

        # Plots
        output_img_path = f'5_Plots/{dataset["path"]}/xgb'
        os.makedirs(output_img_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        var_imp_xgb.head(15).plot(kind="bar", x="Feature", y="Importance", legend=False)
        plt.title(f"XGBoost: Variable Importance ({key})")
        plt.xlabel("Feature"); plt.ylabel("Gain Importance")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(f'{output_img_path}/var_importance_{key}.png')
        plt.close()

        # RMSE
        train_pred = xgb_model.predict(train_x_clean)
        test_pred  = xgb_model.predict(test_x_clean)
        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
        test_rmse  = np.sqrt(mean_squared_error(test_y, test_pred))

        results_xgb_df.loc[key] = [train_rmse, test_rmse, time_taken]

        # Predictions
        pd.DataFrame(train_pred, columns=["Prediction"]).to_pickle(f'{output_data_path}/training_predictions_xgb_{key}')
        pd.DataFrame(test_pred,  columns=["Prediction"]).to_pickle(f'{output_data_path}/test_predictions_xgb_{key}')

    output_model_path = f'4_Model_results/{dataset["path"]}/xgb'
    os.makedirs(output_model_path, exist_ok=True)
    results_xgb_df.to_csv(f'{output_model_path}/xgb_model_comparison.csv', index=False)

print("Done.")
