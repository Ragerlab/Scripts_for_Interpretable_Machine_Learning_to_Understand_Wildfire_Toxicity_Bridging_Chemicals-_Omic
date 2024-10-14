import pandas as pd
import numpy as np
import os
import re

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Read in gene expression data
dat_all = pd.read_csv("Data_inputs/3_Omic_measurements/022821_NormCounts_pslog2_RUV.csv")
samp = pd.read_csv("Data_inputs/3_Omic_measurements/022821_SampleInfo_SamplesIncluded.csv")
degs = pd.read_csv("Data_inputs/3_Omic_measurements/DEGs.csv")
degs.rename(columns={degs.columns[10]: 'sig'}, inplace=True)

# Subset to the 4hr timepoint and get mouse IDs
samp_four = samp.loc[samp['Timepoint'] == '4h']
IDs = samp_four['MouseID']

# Extract Mouse IDS
dat_columns = dat_all.columns
dat_prefixes = dat_columns.str.split('_').str[0]

# Filter columns that match the IDs
matched_columns = dat_columns[dat_prefixes.isin(IDs)]

# Subset the 'dat' DataFrame to only include the matched columns
dat = dat_all[['Genes'] + list(matched_columns)]

# Remove LPS from DEGs list 
degs_filtered = degs[~degs['Exposure_Condition'].str.contains('LPS', na=False)]

# Get DEG names
genes = degs_filtered[['BioSpyder_Identifier', degs.columns[10]]]
gene_sub = genes.loc[genes['sig'] == 'Yes']
deg_genes = gene_sub['BioSpyder_Identifier']

# Subset column to only contain DEGS
dat_deg= dat[dat['Genes'].isin(deg_genes)]

# Transpose
dat_deg = np.transpose(dat_deg)

# Set the first row as column names
dat_deg.columns = dat_deg.iloc[0]  
dat_deg = dat_deg[1:]  

# Move the 'M1_', 'M2_', etc. prefix from the beginning to the end of the row labels in dat_deg
def move_prefix_to_suffix(index_name):
    match = re.match(r'(M\d+_)(.*)', index_name)
    if match:
        prefix, rest = match.groups()
        return rest + '_' + prefix[:-1]  # Removing the trailing underscore from prefix
    return index_name  # Return as is if no match

# Apply the function 
dat_deg.index = dat_deg.index.map(move_prefix_to_suffix)

# Replace 'flame' with 'flaming' and 'smolder' with 'smoldering' 
dat_deg.index = dat_deg.index.str.replace('Flame', 'Flaming', case=False)
dat_deg.index = dat_deg.index.str.replace('Smolder', 'Smoldering', case=False)

# Add in injury protein column
injury_df = pd.read_pickle("Data_inputs/2_Chemical_measurements/Injury_df")
prot = injury_df['Injury_Protein']
dat_deg = dat_deg.join(prot)

# Load in data split from chemical data
train_x_chem = pd.read_pickle("Data_inputs/2_Chemical_measurements/train_x")
test_x_chem = pd.read_pickle("Data_inputs/2_Chemical_measurements/test_x")

# Subset gene expression data
train_x = dat_deg.loc[dat_deg.index.isin(train_x_chem.index)]
train_y = train_x['Injury_Protein']
test_x = dat_deg.loc[dat_deg.index.isin(test_x_chem.index)]
test_y = test_x['Injury_Protein']

# Drop injury protein from training
train_x = train_x.drop('Injury_Protein', axis = 1)
test_x = test_x.drop('Injury_Protein', axis = 1)

# Save data splits for downstream use
dat_deg.to_pickle('Data_inputs/3_Omic_measurements/dat_deg')
train_x.to_pickle("Data_inputs/3_Omic_measurements/train_x")
train_y.to_pickle("Data_inputs/3_Omic_measurements/train_y")
test_y.to_pickle("Data_inputs/3_Omic_measurements/test_y")
test_x.to_pickle("Data_inputs/3_Omic_measurements/test_x")
