import pandas as pd
import numpy as np
import os
import re

# Set working directory
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Read in gene expression data
dat_all = pd.read_csv("1_Data_inputs/3_Omic_measurements/022821_NormCounts_pslog2_RUV.csv")
samp = pd.read_csv("1_Data_inputs/3_Omic_measurements/022821_SampleInfo_SamplesIncluded.csv")
degs = pd.read_csv("1_Data_inputs/3_Omic_measurements/DEGs.csv")
degs.rename(columns={degs.columns[10]: 'sig'}, inplace=True)

# Subset to the 4hr timepoint and get mouse IDs
samp_four = samp.loc[samp['Timepoint'] == '4h']
IDs = samp_four['MouseID']

# Extract Mouse IDs
dat_columns = dat_all.columns
dat_prefixes = dat_columns.str.split('_').str[0]

# Filter columns that match the IDs
matched_columns = dat_columns[dat_prefixes.isin(IDs)]

# Subset the 'dat' DataFrame to only include the matched columns
dat = dat_all[['Genes'] + list(matched_columns)]

# Remove rows with 'LPS' in DEGs
degs_filtered = degs[~degs['Exposure_Condition'].str.contains('LPS', na=False)]

# Get DEG names
genes = degs_filtered[['BioSpyder_Identifier', degs.columns[10]]]
gene_sub = genes.loc[genes['sig'] == 'Yes']
deg_genes = gene_sub['BioSpyder_Identifier']

# Subset columns to only contain DEGs
dat_deg = dat[dat['Genes'].isin(deg_genes)]

# Define a formatting function that also removes rows containing 'LPS' in row indices
def format_dataframe(df):
    # Transpose
    df = df.set_index('Genes').transpose()

    # Remove rows with 'LPS' in the index
    df = df[~df.index.str.contains('LPS', na=False)]

    # Move the 'M1_', 'M2_', etc. prefix from the beginning to the end of the row labels
    def move_prefix_to_suffix(index_name):
        match = re.match(r'(M\d+_)(.*)', index_name)
        if match:
            prefix, rest = match.groups()
            return rest + '_' + prefix[:-1]  # Removing the trailing underscore from prefix
        return index_name  # Return as is if no match

    # Apply the function 
    df.index = df.index.map(move_prefix_to_suffix)

    # Replace 'Flame' with 'Flaming' and 'Smolder' with 'Smoldering' 
    df.index = df.index.str.replace('Flame', 'Flaming', case=False)
    df.index = df.index.str.replace('Smolder', 'Smoldering', case=False)

    # Add injury protein column
    injury_df = pd.read_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_Injury_df")
    prot = injury_df['Injury_Protein']
    df = df.join(prot)

    return df

# Format both dat and dat_deg
dat = format_dataframe(dat)
dat_deg = format_dataframe(dat_deg)

# Save full datasets
dat.to_pickle('3_Data_intermediates/3_Omic_measurements/dat_full.pkl')
dat_deg.to_pickle('3_Data_intermediates/3_Omic_measurements/dat_deg.pkl')

# Load chemical data splits
train_x_chem = pd.read_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_train_x")
test_x_chem = pd.read_pickle("3_Data_intermediates/2_Chemical_measurements/Chem_test_x")

# Subset gene expression data (dat) and DEGs (dat_deg) for training and testing based on chemical data indices
train_x_dat = dat.loc[dat.index.isin(train_x_chem.index)]
test_x_dat = dat.loc[dat.index.isin(test_x_chem.index)]

train_x_deg = dat_deg.loc[dat_deg.index.isin(train_x_chem.index)]
test_x_deg = dat_deg.loc[dat_deg.index.isin(test_x_chem.index)]

# Extract labels for training and testing (assuming 'Injury_Protein' is the label column)
train_y_dat = train_x_dat['Injury_Protein']
test_y_dat = test_x_dat['Injury_Protein']

train_y_deg = train_x_deg['Injury_Protein']
test_y_deg = test_x_deg['Injury_Protein']

# Drop 'Injury_Protein' column from features
train_x_dat = train_x_dat.drop('Injury_Protein', axis=1)
test_x_dat = test_x_dat.drop('Injury_Protein', axis=1)

train_x_deg = train_x_deg.drop('Injury_Protein', axis=1)
test_x_deg = test_x_deg.drop('Injury_Protein', axis=1)

# Save training and testing splits for both dat and dat_deg
train_x_dat.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_train_x")
test_x_dat.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_test_x")
train_y_dat.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_train_y")
test_y_dat.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_test_y")

train_x_deg.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_train_x_deg")
test_x_deg.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_test_x_deg")
train_y_deg.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_train_y_deg")
test_y_deg.to_pickle("3_Data_intermediates/3_Omic_measurements/Omic_test_y_deg")

# Merged for combined analysis
train_x_comb = pd.merge(train_x_deg, train_x_chem, left_index = True, right_index = True)
train_y_comb = train_y_deg
test_x_comb = pd.merge(test_x_deg, test_x_chem, left_index = True, right_index = True)
test_y_comb = test_y_deg

# Save combined form
train_x_comb.to_pickle("3_Data_intermediates/4_ChemOmics_measurements/Comb_train_x")
test_x_comb.to_pickle("3_Data_intermediates/4_ChemOmics_measurements/Comb_test_x")
train_y_comb.to_pickle("3_Data_intermediates/4_ChemOmics_measurements/Comb_train_y")
test_y_comb.to_pickle("3_Data_intermediates/4_ChemOmics_measurements/Comb_test_y")