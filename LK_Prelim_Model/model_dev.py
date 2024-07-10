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
import time
import os
import re
from gplearn.genetic import SymbolicRegressor
import graphviz
import feyn
from feyn.plots import plot_regression
from feyn.plots import plot_residuals

os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# Read in and format mouse tox data
tox = pd.read_excel("LK_Prelim_Model/ChemistrywTox_MouseMap_042821.xlsx", sheet_name=2)
# neutro = tox.rename(columns={"Exposure...1": "Exposure"})
# neutro = neutro[(neutro["Exposure"] != "LPS") & (neutro["Exposure"] != "Saline")]
# neutro["Link"] = neutro["Exposure"] + "_" + neutro["MouseID"]
# neutro = neutro[["Exposure", "Link", "Neutrophil"]]

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
np.random.seed(17)
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

# Pull out the top 4 components
pcs_retain = pca_scores[:,0:4]
train_x_pca = pd.DataFrame(pcs_retain, columns=['PC1', 'PC2', 'PC3', 'PC4']) # Set column names

# Set the row names
train_x_pca.index = train_x.index

# Standardize the test data using the same scaler fitted on the training data
test_x_scaled = scaler.transform(test_x)

# Transform test data to principal components using the same PCA model
test_pca_scores = pca.transform(test_x_scaled)

# Pull out the top 3 components for test data
pcs_retain_test = test_pca_scores[:, 0:4]
test_x_pca = pd.DataFrame(pcs_retain_test, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Set the row names for test data
test_x_pca.index = test_x.index

# Reduce dimensionality using Lasso
# Perform Lasso with cross-validation to select the best alpha (lambda)
lasso_cv = LassoCV(cv=3, max_iter = 100000).fit(train_x, train_y)

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

# Create dictionary containing full input, PCA-reduced input, and Lasso-reduced input
train_input_dict = {'Full': train_x, 'PCA': train_x_pca, 'Lasso': train_x_lasso}
test_input_dict = {'Full': test_x, 'PCA': test_x_pca, 'Lasso': test_x_lasso}

# Convert dictionary keys to a list
keys = list(train_input_dict.keys())

# Initialize a DataFrame to store results
results_rf_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Iterate through inputs and run random forest model
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_test = test_input_dict[key]

    # Model parameters
    rf_model = RandomForestRegressor(n_estimators=10000, random_state=17)

    # Run and time model
    start_time = time.time()
    rf_model.fit(df_train, train_y)
    end_time = time.time()
    time_taken = end_time - start_time

    # Extract variable importance
    importances = rf_model.feature_importances_
    var_imp_rf_injury = pd.DataFrame({"Feature": df_train.columns, "Importance": importances})
    var_imp_rf_injury = var_imp_rf_injury.sort_values("Importance", ascending=False)

    # Plot variable importance
    # plt.figure(figsize=(10, 6))
    # var_imp_rf_injury.plot(kind="bar", x="Feature", y="Importance")
    # plt.title("Variable Importance Plot")
    # plt.xlabel("Feature")
    # plt.ylabel("Importance")
    # plt.show()

    # Get training data RMSE 
    train_pred = rf_model.predict(df_train)
    train_rmse = root_mean_squared_error(train_y, train_pred)

    # Apply model to test set
    test_pred = rf_model.predict(df_test)
    test_rmse = root_mean_squared_error(test_y, test_pred)

    # Store results in DataFrame
    results_rf_df .loc[key] = [train_rmse, test_rmse, time_taken]
results_rf_df 

# Run SRA using pysr
# Define function to clean up names for pysr
def clean_expression(expr):
    # Remove unwanted characters and spaces
    expr = re.sub(r'\s+', '', expr)
    
    # Replace multiple adjacent operators
    expr = re.sub(r'\++', '+', expr)
    expr = re.sub(r'--', '+', expr)
    expr = re.sub(r'-\+', '-', expr)
    expr = re.sub(r'\+-', '-', expr)
    
    # Ensure all variables and numbers are correctly formatted
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', expr)
    
    return expr

# Clean names 
train_x = clean_column_names(train_x)
train_x_pca = clean_column_names(train_x_pca)
train_x_lasso = clean_column_names(train_x_lasso)
test_x = clean_column_names(test_x)
test_x_pca = clean_column_names(test_x_pca)
test_x_lasso = clean_column_names(test_x_lasso)

# Initialize a DataFrame to store results
results_pysr_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Iterate through relevant inputs
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_test = test_input_dict[key]

    # Define parameters - https://astroautomata.com/PySR/api/
    default_pysr_params = dict(
        populations = 15, # default number
        population_size = 33, #default number
        model_selection = "best",
        parsimony = 0.0032 # Multiplicative factor for how much to punish complexity - default value
    )

    # Define loss function - note triple quotes in julia allows line breaks 
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
    # Set up model 
    discovered_model = pysr.PySRRegressor(
        niterations=50,
        # unary_operators=['exp','log'],
        binary_operators=["-", "+", "*", "/"],
        # binary_operators=["-", "+", "*", "/", "^"],
        loss_function=loss_function,
        **default_pysr_params,
        temp_equation_file=True
    )

    # Run and time model 
    start_time = time.time()
    discovered_model.fit(df_train.values, 
                        train_y.values,
                        variable_names=df_train.columns.tolist()
                        )    
    end_time = time.time()
    time_taken = end_time - start_time

    # Save model
    print(discovered_model)
    
    # Pysr Train RMSE 
    y_train = discovered_model.predict(df_train.values)
    train_pysr_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_pysr_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/pysr/pysr_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/pysr/pysr_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # Pysr Test RMSE 
    ypredict = discovered_model.predict(df_test.values)
    test_pysr_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_pysr_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/pysr/pysr_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/pysr/pysr_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_pysr_df.loc[key] = [train_pysr_rmse, test_pysr_rmse, time_taken]
results_pysr_df


# SRA with gplearn
# Initialize a DataFrame to store results
results_gp_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Iterate through relevant inputs
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_test = test_input_dict[key]

    # Set model parameters - https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-regressor
    est_gp = SymbolicRegressor(population_size=5000,
                            generations=10, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=17, 
                            metric = 'rmse', 
                            function_set = ('add', 'sub', 'mul', 'div'))

    # Run and time model 
    start_time = time.time()
    est_gp.fit(df_train, train_y)
    end_time = time.time()
    time_taken = end_time - start_time

    # Print final equation 
    print(est_gp._program)


    # Plot final tree dot_data = est_gp._program.export_graphviz()
    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    file_name = f'gplearn/images/ex1_child_{keys[i]}'
    graph.render(file_name, format='png', cleanup=True)
    graph

    # gplearn Train RMSE 
    y_train = est_gp.predict(df_train.values)
    train_gp_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_gp_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/gplearn/gplearn_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/gplearn/gplearn_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # gplearn Test RMSE 
    ypredict = est_gp.predict(df_test.values)
    test_gp_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_gp_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/gplearn/gplearn_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/gplearn/gplearn_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_gp_df.loc[key] = [train_gp_rmse, test_gp_rmse, time_taken]
results_gp_df

# SRA with feyn
# Initialize a DataFrame to store results
results_feyn_df = pd.DataFrame(columns=["Training RMSE", "Test RMSE", "Time Taken"])

# Iterate through relevant inputs
for i in range(len(train_input_dict)):
    # Subset to relevant input
    key = keys[i]
    df_train = train_input_dict[key]
    df_train['Injury_Protein'] = y_train
    df_test = test_input_dict[key]

    # Set up QLattice 
    ql = feyn.QLattice(random_seed = 17)

    # Run and time model 
    start_time = time.time()
    models = ql.auto_run(data=df_train,
                            output_name='Injury_Protein',
                            kind = 'regression',
                            n_epochs = 10,
                            stypes={"operators": '* / + -'}, 
                            # loss_function = 'rmse' # Need to fix this 
                            )
    end_time = time.time()
    time_taken = end_time - start_time

    # Select the best Model
    best = models[0]
    sympy_model = best.sympify(signif=3)
    sympy_model.as_expr()

    # gplearn Train RMSE 
    y_train = best.predict(df_train)
    train_gp_rmse = root_mean_squared_error(train_y, y_train)
    print(f"Training RMSE: {train_gp_rmse:.2f}")

    # Plot training residuals
    file_name = f'images/feyn/feyn_residuals_train_{keys[i]}'
    plot_residuals(train_y, y_train, filename=file_name)

    # Plot training regression plot
    file_name = f'images/feyn/feyn_regression_train_{keys[i]}'
    plot_regression(train_y, y_train, filename=file_name)

    # gplearn Test RMSE 
    ypredict = best.predict(df_test)
    test_gp_rmse = root_mean_squared_error(test_y, ypredict)
    print(f"Testing RMSE: {test_gp_rmse:.2f}")

    # Plot testing residuals
    file_name = f'images/feyn/feyn_residuals_test_{keys[i]}'
    plot_residuals(test_y, ypredict, filename=file_name)

    # Plot testing regression plot
    file_name = f'images/feyn/feyn_regression_test_{keys[i]}'
    plot_regression(test_y, ypredict, filename=file_name)

    # Store results in DataFrame
    results_feyn_df.loc[key] = [train_gp_rmse, test_gp_rmse, time_taken]
results_feyn_df