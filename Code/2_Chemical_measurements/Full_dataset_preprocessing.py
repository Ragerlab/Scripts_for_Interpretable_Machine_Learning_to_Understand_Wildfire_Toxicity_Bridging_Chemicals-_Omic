import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set working directory
os.chdir(r"C:\Users\jrchapp3\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

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

# Apply the log transformation to all columns except the first one ('Injury Protein')
injury_df.iloc[:, 1:] = injury_df.iloc[:, 1:].applymap(lambda x: np.log(x + 1))

# Split columns into 4 groups
columns_per_plot = len(injury_df.columns) // 4
column_groups = [injury_df.columns[i:i + columns_per_plot] for i in range(0, len(injury_df.columns), columns_per_plot)]

# Plotting each group separately
for i, group in enumerate(column_groups, start=1):
    plt.figure(figsize=(10, 6))
    injury_df[group].boxplot()
    plt.title(f'Distribution of Columns Group {i} in Injury DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Log-Transformed Values')
    plt.xticks(rotation=45)
    plt.savefig(f'Images/2_Chemical_measurements/Data_distributions/Concentration_spread{i}.png')

# Remove outlier
injury_df = injury_df.drop('EucalyptusSmoldering_M28', axis=0, errors='ignore')

# Set seed and establish train and test sets
np.random.seed(17)
train_x, test_x, train_y, test_y = train_test_split(injury_df.drop("Injury_Protein", axis=1), injury_df["Injury_Protein"], test_size=0.4)

# Save data splits for downstream use
injury_df.to_pickle("Data_inputs/2_Chemical_measurements/Injury_df")
train_x.to_pickle("Data_inputs/2_Chemical_measurements/train_x")
train_y.to_pickle("Data_inputs/2_Chemical_measurements/train_y")
test_x.to_pickle("Data_inputs/2_Chemical_measurements/test_x")
