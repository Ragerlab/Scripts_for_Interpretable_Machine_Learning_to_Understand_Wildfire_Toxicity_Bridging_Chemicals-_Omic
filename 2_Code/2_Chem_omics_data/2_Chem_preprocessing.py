import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

INPUT_FILE = "1_Data_inputs/2_Chemical_measurements/ChemistrywTox_MouseMap_042821_mw.xlsx"
UNICODE_SPACES_PATTERN = r"[\u00A0\u2007\u202F]"  # NBSP / figure space / narrow NBSP
FALLBACK_EPS = 1e-30

def _sanitize_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(UNICODE_SPACES_PATTERN, "", regex=True)
         .str.strip()
         .str.replace(",", "", regex=False),
        errors="coerce"
    )

def convert_conc_to_mol_per_L(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    # Convert (mass/volume) -> g/L
    unit_to_g_per_L = {"ng/ul": 1e-3, "ng/ml": 1e-6, "ug/ml": 1e-3}

    df = df.copy()

    unit_norm = (
        df["Units"].astype(str).str.strip().str.lower()
        .str.replace("µ", "u").str.replace("μ", "u")
        .str.replace(" ", "")
    )
    df["Molecular_weight"] = _sanitize_to_numeric(df["Molecular_weight"])

    g_per_L_factor = unit_norm.map(unit_to_g_per_L)

    for c in value_cols:
        df[c] = _sanitize_to_numeric(df[c])
        df[c] = (df[c] * g_per_L_factor) / df["Molecular_weight"]  # mol/L

    return df

def log_with_eps_per_col(df: pd.DataFrame, cols: list[str], fallback_eps: float = FALLBACK_EPS) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        x = df[c].to_numpy(dtype=float)
        pos = x[np.isfinite(x) & (x > 0)]
        eps = (pos.min() / 2.0) if pos.size else fallback_eps
        df[c] = np.log10(x + eps)

    return df

# Read in and format mouse tox data
tox = pd.read_excel(INPUT_FILE, sheet_name=2)

# Isolate injury protein marker (outcome var) from tox dataset
injury = tox.rename(columns={"Exposure...1": "Exposure"})
injury = injury[(injury["Exposure"] != "LPS") & (injury["Exposure"] != "Saline")]
injury["Link"] = injury["Exposure"] + "_" + injury["MouseID"]
injury = injury[["Exposure", "Link", "Injury_Protein"]]

# Read in and format burn chemistry data (predictor vars)
chem_raw = pd.read_excel(INPUT_FILE, sheet_name=1)
exps = [col for col in chem_raw.columns if "Flaming" in col or "Smoldering" in col]

# Convert concentrations to mol/L
chem_raw = convert_conc_to_mol_per_L(chem_raw, value_cols=exps)

chem = chem_raw[["Chemical"] + exps].set_index("Chemical").T.reset_index().rename(columns={"index": "Exposure"})

# Merge injury protein markers with chemistry data
injury_df = pd.merge(injury, chem, on="Exposure", how="left")
injury_df = injury_df.set_index("Link")
injury_df = injury_df.select_dtypes(include=["number"])

# Epsilon log-transform predictors only (not Injury_Protein)
predictor_cols = [c for c in injury_df.columns if c != "Injury_Protein"]
injury_df = log_with_eps_per_col(injury_df, cols=predictor_cols, fallback_eps=FALLBACK_EPS)

# Split columns into 4 groups for plotting
columns_per_plot = max(1, len(injury_df.columns) // 4)
column_groups = [injury_df.columns[i:i + columns_per_plot] for i in range(0, len(injury_df.columns), columns_per_plot)]

for i, group in enumerate(column_groups, start=1):
    plt.figure(figsize=(10, 6))
    injury_df[group].boxplot()
    plt.title(f'Distribution of Columns Group {i} in Injury DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Log(x + eps) Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'5_Plots/2_Chemical_measurements/Data_distributions/Concentration_spread{i}.png')
    plt.close()

# Remove outlier
injury_df = injury_df.drop('EucalyptusSmoldering_M28', axis=0, errors='ignore')

# Set seed and establish train and test sets
np.random.seed(17)
train_x, test_x, train_y, test_y = train_test_split(
    injury_df.drop("Injury_Protein", axis=1),
    injury_df["Injury_Protein"],
    test_size=0.4
)

# Save data splits for downstream use
injury_df.to_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_Injury_df")
train_x.to_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_train_x")
train_y.to_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_train_y")
test_x.to_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_test_x")
test_y.to_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_test_y")
