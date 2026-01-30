import os
import pandas as pd
import pickle
import sympy as sp
from scipy.integrate import nquad, IntegrationWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings

# -------------------------------------------------------------------
# Set working directory
# -------------------------------------------------------------------
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# -------------------------------------------------------------------
# Load simulated data
# -------------------------------------------------------------------
with open('3_Data_intermediates/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)

# -------------------------------------------------------------------
# Sympy locals 
# -------------------------------------------------------------------
SYMPY_LOCALS = {
    "myfunction": lambda x: x,
    "sqrt": sp.sqrt,
    "log": sp.log,
}

# -------------------------------------------------------------------
# Numerical integration (
# -------------------------------------------------------------------
def integrate_over_all_variables(partial_derivative, all_symbols, ranges):
    try:
        real_part = sp.re(partial_derivative)
        func = sp.lambdify(all_symbols, real_part, 'numpy')

        def integrand(*args):
            try:
                result = func(*args)
                result = float(result)
                if np.isnan(result) or np.isinf(result):
                    return 0
                return result
            except Exception:
                return 0

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            result, _ = nquad(integrand, ranges)

        return result
    except Exception as e:
        print(f"Error during numerical integration setup: {str(e)}")
        return 0

# -------------------------------------------------------------------
# Paths / operator-level subdirectories
# -------------------------------------------------------------------
results_directory  = "4_Model_results/1_Simulated_data/pysr/Variable_importance"
images_directory   = "5_Plots/1_Simulated_data/pysr"
subdirectories     = ["low", "med", "high"]

os.makedirs(results_directory, exist_ok=True)
os.makedirs(images_directory, exist_ok=True)

# -------------------------------------------------------------------
# Compute dataset-specific ranges 
# -------------------------------------------------------------------
dataset_ranges = {}
for dataset_key, data in sim_dict.items():
    x = data.drop("Response", axis=1)
    dataset_ranges[dataset_key] = {col: (x[col].min(), x[col].max()) for col in x.columns}

# -------------------------------------------------------------------
# Helper: load HOF for a specific (operator_level, dataset_key)
# -------------------------------------------------------------------
base_hof_directory = r"4_Model_results/1_Simulated_data/pysr/HOF_all_iterations"

def load_hof_all_iterations(operator_level, dataset_key):
    hof_dir = os.path.join(base_hof_directory, operator_level, dataset_key)

    if not os.path.isdir(hof_dir):
        raise FileNotFoundError(f"Missing HOF iteration directory: {hof_dir}")

    files = [
        fn for fn in os.listdir(hof_dir)
        if fn.lower().endswith(".csv") and "hall_of_fame_iteration_" in fn.lower()
    ]
    if len(files) == 0:
        raise FileNotFoundError(f"No iteration CSVs found in: {hof_dir}")

    dfs = []
    for fn in files:
        path = os.path.join(hof_dir, fn)
        df = pd.read_csv(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["Dataset"] = dataset_key
    combined["Directory"] = operator_level
    return combined

# -------------------------------------------------------------------
# MAIN: do Part 1 (partials+integration) and Part 2 (VI) per dataset
# -------------------------------------------------------------------
for operator_level in subdirectories:
    print(f"Operator level: {operator_level}")

    for dataset_key in sim_dict.keys():
        print(f"\n--- Dataset: {dataset_key} ---")

        # ----------------------------
        # Load HOF for this dataset/level
        # ----------------------------
        hof_df = load_hof_all_iterations(operator_level, dataset_key)


        # Add identifiers (handy in outputs)
        hof_df = hof_df.copy()
        hof_df["Dataset"] = dataset_key
        hof_df["Directory"] = operator_level

        # Use dataset-specific ranges
        chemical_ranges = dataset_ranges[dataset_key]

        # ----------------------------
        # PART 1: partial derivatives + integration
        # ----------------------------
        chems = set()
        for equation in hof_df["equation"].astype(str):
            try:
                expr = sp.sympify(equation, locals=SYMPY_LOCALS)
                chems.update(expr.free_symbols)
            except Exception:
                continue
        chems = list(chems)

        final_results_df = pd.DataFrame(
            columns=[
                "chem",
                "equation",
                "parital derivative w/ respect to chem",
                "integrated_derivative",
            ]
        )

        for j, chem in enumerate(chems):
            print(f"  j_{j}: {chem}")

            subset_df = hof_df[
                hof_df["equation"].astype(str).str.contains(rf"\b{chem}\b", na=False)
            ].copy()

            uniq_eqs = subset_df["equation"].astype(str).unique()

            results_df = pd.DataFrame(
                columns=[
                    "chem",
                    "equation",
                    "parital derivative w/ respect to chem",
                    "integrated_derivative",
                ]
            )

            for k, equation_str in enumerate(uniq_eqs):
                print(f"    k_{k}")

                try:
                    equation_sympy = sp.sympify(equation_str, locals=SYMPY_LOCALS)
                    partial_derivative = sp.diff(equation_sympy, chem)

                    all_symbols = list(partial_derivative.free_symbols)
                    if chem not in all_symbols:
                        all_symbols.append(chem)

                    ranges = []
                    for sym in all_symbols:
                        sym_str = str(sym)
                        if sym_str in chemical_ranges:
                            ranges.append(chemical_ranges[sym_str])
                        else:
                            raise ValueError(f"Range for symbol {sym_str} not found in dataset {dataset_key}.")

                    integrated_derivative = integrate_over_all_variables(
                        partial_derivative, all_symbols, ranges
                    )

                except Exception:
                    partial_derivative = "Error"
                    integrated_derivative = "Error"

                results_df.loc[len(results_df)] = {
                    "chem": chem,
                    "equation": equation_str,
                    "parital derivative w/ respect to chem": partial_derivative,
                    "integrated_derivative": integrated_derivative,
                }

            # merge derivative results back onto the subset rows
            results_subset = pd.merge(subset_df, results_df, how="left", on="equation")
            final_results_df = pd.concat([final_results_df, results_subset], ignore_index=True)

        out_partials = os.path.join(results_directory, f"partial_deriv_{operator_level}_{dataset_key}.csv")
        final_results_df.to_csv(out_partials, index=False)
        print(f"Saved partial derivatives: {out_partials}")

        # ----------------------------
        # PART 2: variable importance (same method, per dataset)
        # ----------------------------
        results_df = pd.read_csv(out_partials)

        var_importance_list = []
        chems_in_results = results_df["chem"].unique()

        for chem in chems_in_results:
            chem_rows = results_df[results_df["chem"] == chem].copy()
            chem_uniq = chem_rows.drop_duplicates().copy()

            chem_uniq["integrated_derivative"] = pd.to_numeric(
                chem_uniq["integrated_derivative"], errors="coerce"
            )
            sum_integrated_derivative = chem_uniq["integrated_derivative"].sum()

            chem_str = str(chem)
            if chem_str not in chemical_ranges:
                continue

            min_value, max_value = chemical_ranges[chem_str]
            range_value = max_value - min_value

            # minimal safety for zero-range (constant variable)
            if range_value == 0:
                var_importance = np.nan
            else:
                var_importance = sum_integrated_derivative / range_value

            var_importance_list.append([
                chem_str,
                var_importance,
                len(chem_rows),
                len(chem_uniq)
            ])

        var_importance_df = pd.DataFrame(
            var_importance_list,
            columns=["chem", "var_importance", "tot_instances", "uniq_instances"]
        ).sort_values(by="var_importance", ascending=False)

        out_vi = os.path.join(results_directory, f"variable_importance_{operator_level}_{dataset_key}.csv")
        var_importance_df.to_csv(out_vi, index=False)
        print(f"Saved variable importance: {out_vi}")
import os
import pandas as pd
import pickle
import sympy as sp
from scipy.integrate import nquad, IntegrationWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings

# -------------------------------------------------------------------
# Set working directory
# -------------------------------------------------------------------
os.chdir(r"C:\Users\Jessie PC\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

# -------------------------------------------------------------------
# Load simulated data
# -------------------------------------------------------------------
with open('3_Data_intermediates/1_Simulated_data/sim_dict.pkl', 'rb') as f:
    sim_dict = pickle.load(f)

# -------------------------------------------------------------------
# Sympy locals 
# -------------------------------------------------------------------
SYMPY_LOCALS = {
    "myfunction": lambda x: x,
    "sqrt": sp.sqrt,
    "log": sp.log,
}

# -------------------------------------------------------------------
# Numerical integration (
# -------------------------------------------------------------------
def integrate_over_all_variables(partial_derivative, all_symbols, ranges):
    try:
        real_part = sp.re(partial_derivative)
        func = sp.lambdify(all_symbols, real_part, 'numpy')

        def integrand(*args):
            try:
                result = func(*args)
                result = float(result)
                if np.isnan(result) or np.isinf(result):
                    return 0
                return result
            except Exception:
                return 0

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            result, _ = nquad(integrand, ranges)

        return result
    except Exception as e:
        print(f"Error during numerical integration setup: {str(e)}")
        return 0

# -------------------------------------------------------------------
# Paths / operator-level subdirectories
# -------------------------------------------------------------------
results_directory  = "4_Model_results/1_Simulated_data/pysr/Variable_importance"
images_directory   = "5_Plots/1_Simulated_data/pysr"
subdirectories     = ["low", "med", "high"]

os.makedirs(results_directory, exist_ok=True)
os.makedirs(images_directory, exist_ok=True)

# -------------------------------------------------------------------
# Compute dataset-specific ranges 
# -------------------------------------------------------------------
dataset_ranges = {}
for dataset_key, data in sim_dict.items():
    x = data.drop("Response", axis=1)
    dataset_ranges[dataset_key] = {col: (x[col].min(), x[col].max()) for col in x.columns}

# -------------------------------------------------------------------
# Helper: load HOF for a specific (operator_level, dataset_key)
# -------------------------------------------------------------------
base_hof_directory = r"4_Model_results/1_Simulated_data/pysr/HOF_all_iterations"

def load_hof_all_iterations(operator_level, dataset_key):
    hof_dir = os.path.join(base_hof_directory, operator_level, dataset_key)

    if not os.path.isdir(hof_dir):
        raise FileNotFoundError(f"Missing HOF iteration directory: {hof_dir}")

    files = [
        fn for fn in os.listdir(hof_dir)
        if fn.lower().endswith(".csv") and "hall_of_fame_iteration_" in fn.lower()
    ]
    if len(files) == 0:
        raise FileNotFoundError(f"No iteration CSVs found in: {hof_dir}")

    dfs = []
    for fn in files:
        path = os.path.join(hof_dir, fn)
        df = pd.read_csv(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['loss'] < 17]
    combined["Dataset"] = dataset_key
    combined["Directory"] = operator_level
    return combined

# -------------------------------------------------------------------
# MAIN: do Part 1 (partials+integration) and Part 2 (VI) per dataset
# -------------------------------------------------------------------
for operator_level in subdirectories:
    print(f"Operator level: {operator_level}")

    for dataset_key in sim_dict.keys():
        print(f"\n--- Dataset: {dataset_key} ---")

        # ----------------------------
        # Load HOF for this dataset/level
        # ----------------------------
        hof_df = load_hof_all_iterations(operator_level, dataset_key)


        # Add identifiers (handy in outputs)
        hof_df = hof_df.copy()
        hof_df["Dataset"] = dataset_key
        hof_df["Directory"] = operator_level

        # Use dataset-specific ranges
        chemical_ranges = dataset_ranges[dataset_key]

        # ----------------------------
        # PART 1: partial derivatives + integration
        # ----------------------------
        chems = set()
        for equation in hof_df["equation"].astype(str):
            try:
                expr = sp.sympify(equation, locals=SYMPY_LOCALS)
                chems.update(expr.free_symbols)
            except Exception:
                continue
        chems = list(chems)

        final_results_df = pd.DataFrame(
            columns=[
                "chem",
                "equation",
                "parital derivative w/ respect to chem",
                "integrated_derivative",
            ]
        )

        for j, chem in enumerate(chems):
            print(f"  j_{j}: {chem}")

            subset_df = hof_df[
                hof_df["equation"].astype(str).str.contains(rf"\b{chem}\b", na=False)
            ].copy()

            uniq_eqs = subset_df["equation"].astype(str).unique()

            results_df = pd.DataFrame(
                columns=[
                    "chem",
                    "equation",
                    "parital derivative w/ respect to chem",
                    "integrated_derivative",
                ]
            )

            for k, equation_str in enumerate(uniq_eqs):
                print(f"    k_{k}")

                try:
                    equation_sympy = sp.sympify(equation_str, locals=SYMPY_LOCALS)
                    partial_derivative = sp.diff(equation_sympy, chem)

                    all_symbols = list(partial_derivative.free_symbols)
                    if chem not in all_symbols:
                        all_symbols.append(chem)

                    ranges = []
                    for sym in all_symbols:
                        sym_str = str(sym)
                        if sym_str in chemical_ranges:
                            ranges.append(chemical_ranges[sym_str])
                        else:
                            raise ValueError(f"Range for symbol {sym_str} not found in dataset {dataset_key}.")

                    integrated_derivative = integrate_over_all_variables(
                        partial_derivative, all_symbols, ranges
                    )

                except Exception:
                    partial_derivative = "Error"
                    integrated_derivative = "Error"

                results_df.loc[len(results_df)] = {
                    "chem": chem,
                    "equation": equation_str,
                    "parital derivative w/ respect to chem": partial_derivative,
                    "integrated_derivative": integrated_derivative,
                }

            # merge derivative results back onto the subset rows
            results_subset = pd.merge(subset_df, results_df, how="left", on="equation")
            final_results_df = pd.concat([final_results_df, results_subset], ignore_index=True)

        out_partials = os.path.join(results_directory, f"partial_deriv_{operator_level}_{dataset_key}.csv")
        final_results_df.to_csv(out_partials, index=False)
        print(f"Saved partial derivatives: {out_partials}")

        # ----------------------------
        # PART 2: variable importance (same method, per dataset)
        # ----------------------------
        results_df = pd.read_csv(out_partials)

        var_importance_list = []
        chems_in_results = results_df["chem"].unique()

        for chem in chems_in_results:
            chem_rows = results_df[results_df["chem"] == chem].copy()
            chem_uniq = chem_rows.drop_duplicates().copy()

            chem_uniq["integrated_derivative"] = pd.to_numeric(
                chem_uniq["integrated_derivative"], errors="coerce"
            )
            sum_integrated_derivative = chem_uniq["integrated_derivative"].sum()

            chem_str = str(chem)
            if chem_str not in chemical_ranges:
                continue

            min_value, max_value = chemical_ranges[chem_str]
            range_value = max_value - min_value

            # minimal safety for zero-range (constant variable)
            if range_value == 0:
                var_importance = np.nan
            else:
                var_importance = sum_integrated_derivative / range_value

            var_importance_list.append([
                chem_str,
                var_importance,
                len(chem_rows),
                len(chem_uniq)
            ])

        var_importance_df = pd.DataFrame(
            var_importance_list,
            columns=["chem", "var_importance", "tot_instances", "uniq_instances"]
        ).sort_values(by="var_importance", ascending=False)

        out_vi = os.path.join(results_directory, f"variable_importance_{operator_level}_{dataset_key}.csv")
        var_importance_df.to_csv(out_vi, index=False)
        print(f"Saved variable importance: {out_vi}")
