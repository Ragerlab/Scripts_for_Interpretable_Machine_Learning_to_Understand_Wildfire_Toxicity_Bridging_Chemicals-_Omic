import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI\Albumin_runs")

# Read in and format mouse tox data
tox = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=2)

# Isolate injury Albumin marker (outcome var) from tox dataset
injury = tox.rename(columns={"Exposure...1": "Exposure"})
injury = injury[(injury["Exposure"] != "LPS") & (injury["Exposure"] != "Saline")]
injury["Link"] = injury["Exposure"] + "_" + injury["MouseID"]
injury = injury[["Exposure", "Link", "Injury_Albumin"]]

# Read in and format burn chemistry data (predictor vars)
chem = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=1)
exps = [col for col in chem.columns if "Flaming" in col or "Smoldering" in col]
chem = chem[["Chemical"] + exps]
chem = chem.set_index("Chemical").T
chem = chem.reset_index()
chem = chem.rename(columns={"index": "Exposure"})

# Merge injury Albumin markers with chemistry data
injury_df = pd.merge(injury, chem, on="Exposure", how="left")
injury_df = injury_df.set_index("Link")
injury_df = injury_df.select_dtypes(include=["number"])

# Apply the log transformation to all columns except the first one ('Injury Albumin')
injury_df.iloc[:, 1:] = injury_df.iloc[:, 1:].applymap(lambda x: np.log(x + 1))

# Remove outlier
injury_df = injury_df.drop('EucalyptusSmoldering_M28', axis=0, errors='ignore')

# Set seed and establish train and test sets
np.random.seed(17)
train_x, test_x, train_y, test_y = train_test_split(injury_df.drop("Injury_Albumin", axis=1), injury_df["Injury_Albumin"], test_size=0.4)

# Save data splits for downstream use
injury_df.to_pickle("Data_inputs/2_Chemical_measurements/Injury_df")
train_x.to_pickle("Data_inputs/2_Chemical_measurements/train_x")
train_y.to_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_x.to_pickle("Data_inputs/2_Chemical_measurements/test_x")
test_y.to_pickle("Data_inputs/2_Chemical_measurements/test_y")
