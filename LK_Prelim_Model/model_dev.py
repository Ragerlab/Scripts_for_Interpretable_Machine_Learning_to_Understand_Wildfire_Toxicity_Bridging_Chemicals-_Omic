import pysr
import sympy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt

# Read in and format mouse tox data
tox = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=2)
neutro = neutro[(neutro["Exposure"] != "LPS") & (neutro["Exposure"] != "Saline")]
neutro["Link"] = neutro["Exposure"] + "_" + neutro["MouseID"]
neutro = neutro[["Exposure", "Link", "Neutrophil"]]

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

# Set seed and establish train and test sets
train_x, test_x, train_y, test_y = train_test_split(injury_df.drop("Injury_Protein", axis=1), injury_df["Injury_Protein"], test_size=0.4)

# Reduce dimensionality using PCA
# Subset data and standardize values
scaler = StandardScaler()
pca_sub_scaled = scaler.fit_transform(train_x)

# Create PCA object
pca = PCA(n_components = 10)

# Fit PCA to data excluding protein injury
pca.fit(pca_sub_scaled)

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()

# Transform data to principal components
pca_scores = pca.transform(pca_sub_scaled)

# Scores plot 
plt.figure(figsize=(10, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.8)
# Add labels based on row names in pca_sub_unique
for i, label in enumerate(train_x.index):
    plt.annotate(label, (pca_scores[i, 0], pca_scores[i, 1]))
plt.title('Scores Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Pull out the top 3 components
pcs_retain = pca_scores[:,0:3]
train_x_pca = pd.DataFrame(pcs_retain, columns=['PC1', 'PC2', 'PC3']) # Set column names

# Set the row names
train_x_pca.index = train_x.index

# Reduce dimensionality using Lasso
# Perform Lasso with cross-validation to select the best alpha (lambda)
lasso_cv = LassoCV(cv=3, random_state=17, max_iter = 100000).fit(train_x, train_y)

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


# Train model
rf_model = RandomForestRegressor(n_estimators=500, random_state=17)
rf_model.fit(train_x, train_y)

# Extract variable importance
importances = rf_model.feature_importances_
var_imp_rf_injury = pd.DataFrame({"Feature": train_x.columns, "Importance": importances})
var_imp_rf_injury = var_imp_rf_injury.sort_values("Importance", ascending=False)

# Get training data RMSE 
train_pred = rf_model.predict(train_x)
rmse = root_mean_squared_error(train_y, train_pred)
print(f"Train RMSE: {rmse:.2f}")

# Apply model to test set
test_pred = rf_model.predict(test_x)
rmse = root_mean_squared_error(test_y, test_pred)
print(f"Test RMSE: {rmse:.2f}")

# Plot variable importance
# plt.figure(figsize=(10, 6))
# var_imp_rf_injury.plot(kind="bar", x="Feature", y="Importance")
# plt.title("Variable Importance Plot")
# plt.xlabel("Feature")
# plt.ylabel("Importance")
# plt.show()

default_pysr_params = dict(
    populations = 10,
    model_selection = "best",
)

chem_names = train_x.columns.tolist()
for n,chem in enumerate(chem_names):
    if "-" in chem:
        chem = chem.replace('-','')
    if "," in chem:
        chem = chem.replace(',','')
    if "(" in chem:
        chem = chem.replace('(','')
    if ")" in chem:
        chem = chem.replace(')','')
    if "[" in chem:
        chem = chem.replace('[','')
    if "]" in chem:
        chem = chem.replace(']','')
    if chem == "S":
        chem = "Sulphur"
    if chem == "Si":
        chem = "Silicon"
    chem_names[n] = chem

loss_function = """
function f(tree, dataset::Dataset{T,L}, options) where {T,L}
    ypred, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return L(Inf)
    end
    y = dataset.y
    return sqrt( (1 / length(y)) * sum(i -> (ypred[i] - y[i])^2, eachindex(y)))
end
"""

discovered_model = pysr.PySRRegressor(
    niterations=500,
    # unary_operators=['exp','log'],
    binary_operators=["-", "+", "*", "/"],
    # binary_operators=["-", "+", "*", "/", "^"],
    loss_function=loss_function,
    **default_pysr_params,
)

discovered_model.fit(train_x.values, 
                     train_y.values,
                     variable_names=chem_names
                     )


# print(discovered_model)

# Pysr Train RMSE 
y_train = discovered_model.predict(train_x.values)
pysr_rmse = root_mean_squared_error(train_y, y_train)
print(f"Training RMSE: {pysr_rmse:.2f}")

# for n in range(1,16):
ypredict = discovered_model.predict(test_x.values)
pysr_rmse = root_mean_squared_error(test_y, ypredict)
print(f"Testing RMSE: {pysr_rmse:.2f}")