import pandas as pd
import numpy as np
import pysr
import matplotlib.pyplot as plt 

# https://astroautomata.com/PySR/
# Set random seed for reproducibility
np.random.seed(17)

# Number of data points
n = 500

# Generate random data for 15 chemicals with various distributions and ranges
chem1 = np.random.uniform(1, 10, n)
chem2 = np.random.uniform(2, 5.0, n)
chem3 = np.random.uniform(4, 12, n)
chem4 = np.random.uniform(0.5, 6, n)
chem5 = np.random.uniform(6, 10, n)
chem6 = np.random.uniform(1, 7, n)
chem7 = np.random.normal(loc=3, scale=1, size=n)
chem8 = np.random.normal(loc=5, scale=1.5, size=n)
chem9 = np.random.normal(loc=7, scale=2, size=n)
chem10 = np.random.beta(a=2, b=5, size=n)
chem11 = np.random.beta(a=5, b=2, size=n)
chem12 = np.random.normal(loc=6, scale=2, size=n)
chem13 = np.random.normal(loc=4, scale=1.5, size=n)
chem14 = np.random.lognormal(mean=0, sigma=1, size=n)
chem15 = np.random.lognormal(mean=1, sigma=0.5, size=n)

# Calculate the response
response = (0.5 * chem3 
            + chem3 * chem5 
            + 5*(chem7 / chem12))

# Add Gaussian noise
noise = np.random.normal(0, 2, n)
response_noisy = response + noise

# Compare noisy to actual values 
plt.scatter(response, response_noisy)
plt.xlabel('Actual respsonse')
plt.ylabel('Noisy response')
plt.show()

# Create a DataFrame
data = pd.DataFrame({
    'chem1': chem1,
    'chem2': chem2,
    'chem3': chem3,
    'chem4': chem4,
    'chem5': chem5, 
    'chem6': chem6,
    'chem7': chem7,
    'chem8': chem8,
    'chem9': chem9,
    'chem10': chem10,
    'chem11': chem11,
    'chem12': chem12,
    'chem13': chem13,
    'chem14': chem14,
    'chem15': chem15,
    'Response': response_noisy
})

# Define parameters - https://astroautomata.com/PySR/api/
default_pysr_params = dict(
        populations = 30, # default number
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
        niterations=1000,
        # unary_operators=['exp','log'],
        binary_operators=["-", "+", "*", "/"],
        # binary_operators=["-", "+", "*", "/", "^"],
        loss_function=loss_function,
        **default_pysr_params,
        temp_equation_file=True, 
        warm_start= True
    )

# Set up input and output
x = data.drop('Response', axis = 1)
y = data['Response']

# Run model 
discovered_model.fit(x.values, 
                        y.values,
                        variable_names=x.columns.tolist()
                        )    
print(discovered_model.fit)

# Get top 10 models
equations = pd.DataFrame(discovered_model.equations_)

# Save the DataFrame to a CSV file
file_name = f'Models/Simulated/pysr_HOF.csv'
equations.to_csv(file_name, index=False)

# Compare equation predictions to actual
pred = discovered_model.predict(x)
plt.scatter(response, pred)
plt.xlabel('Equation Output')
plt.ylabel('Model Output')
plt.show()
