import pandas as pd
import numpy as np
import pysr
import matplotlib.pyplot as plt 

# Example from pysr website - https://astroautomata.com/PySR/


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
