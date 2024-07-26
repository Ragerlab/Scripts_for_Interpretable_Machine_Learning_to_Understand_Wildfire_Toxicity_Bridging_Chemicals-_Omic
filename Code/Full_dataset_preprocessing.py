import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Read in and format mouse tox data
tox = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=2)

# Isolate injury protein marker (outcome var) from tox dataset
injury = tox.rename(columns={"Exposure...1": "Exposure"})
injury = injury[(injury["Exposure"] != "LPS") & (injury["Exposure"] != "Saline")]
injury["Link"] = injury["Exposure"] + "_" + injury["MouseID"]
injury = injury[["Exposure", "Link", "Injury_Protein"]]

# Read in and format burn chemistry data (predictor vars)
chem = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=1)
exps = [col for col in chem.columns if "Flaming" in col or "Smoldering" in col]
chem = chem[["Chemical"] + exps]
chem = chem.set_index("Chemical").T
chem = chem.reset_index()
chem = chem.rename(columns={"index": "Exposure"})

# Merge injury protein markers with chemistry data
injury_df = pd.merge(injury, chem, on="Exposure", how="left")
injury_df = injury_df.set_index("Link")
injury_df = injury_df.select_dtypes(include=["number"])

# Check for outliers
injury = injury_df["Injury_Protein"]
inj_mean = injury.mean()
inj_std = injury.std()
low_thresh = inj_mean - 3 * inj_std
up_thresh = inj_mean + 3 * inj_std
injury_df = injury_df[(injury_df["Injury_Protein"] >= low_thresh) & (injury_df["Injury_Protein"] <= up_thresh)]

# Set seed and establish train and test sets
np.random.seed(17)
train_x, test_x, train_y, test_y = train_test_split(injury_df.drop("Injury_Protein", axis=1), injury_df["Injury_Protein"], test_size=0.4)

# Save data splits for downstream use
train_x.to_pickle("Data_inputs/train_x")
train_y.to_pickle("Data_inputs/train_y")
test_x.to_pickle("Data_inputs/test_x")
test_y.to_pickle("Data_inputs/test_y")