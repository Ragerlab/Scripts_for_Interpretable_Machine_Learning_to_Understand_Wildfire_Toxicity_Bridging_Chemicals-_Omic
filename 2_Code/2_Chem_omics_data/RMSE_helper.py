# File: compute_best_hof_and_rmse.py
# Purpose: After PySR training, load the last HOF for each dataset/key, pick the
#          model with the lowest 'score' (fallback to lowest 'loss'), evaluate RMSE.

import os
import re
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# ---------------------------
# Project root & datasets
# ---------------------------
os.chdir(r"C:\Users\jrchapp3\OneDrive - University of North Carolina at Chapel Hill\Symbolic_regression_github\NIH_Cloud_NOSI")

DATASETS = [
    {
        "prefix": "Chem",
        "path": "2_Chemical_measurements",
        "train_y": "3_Data_intermediates/2_Chemical_measurements/Chem_train_y",
        "test_y":  "3_Data_intermediates/2_Chemical_measurements/Chem_test_y",
        "train_input_dict": "3_Data_intermediates/2_Chemical_measurements/train_input_dict.pkl",
        "test_input_dict":  "3_Data_intermediates/2_Chemical_measurements/test_input_dict.pkl",
    },
    {
        "prefix": "Omic",
        "path": "3_Omic_measurements",
        "train_y": "3_Data_intermediates/3_Omic_measurements/Omic_train_y",
        "test_y":  "3_Data_intermediates/3_Omic_measurements/Omic_test_y",
        "train_input_dict": "3_Data_intermediates/3_Omic_measurements/train_input_dict.pkl",
        "test_input_dict":  "3_Data_intermediates/3_Omic_measurements/test_input_dict.pkl",
    },
    {
        "prefix": "Combined",
        "path": "4_ChemOmics_measurements",
        "train_y": "3_Data_intermediates/4_ChemOmics_measurements/Comb_train_y",
        "test_y":  "3_Data_intermediates/4_ChemOmics_measurements/Comb_test_y",
        "train_input_dict": "3_Data_intermediates/4_ChemOmics_measurements/train_input_dict.pkl",
        "test_input_dict":  "3_Data_intermediates/4_ChemOmics_measurements/test_input_dict.pkl",
    },
]

# ---------------------------
# Helpers
# ---------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the cleaning used before PySR so names match the equations."""
    new_cols = []
    for col in df.columns:
        if col == 'S':
            new_cols.append('Sulphur')
        elif col == 'Si':
            new_cols.append('Silicon')
        else:
            cleaned = re.sub(r'\W+', '', col)
            cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', cleaned)
            cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', cleaned)
            if cleaned and cleaned[0].isdigit():
                cleaned = 'var' + cleaned
            new_cols.append(cleaned)
    df = df.copy()
    df.columns = new_cols
    return df

def find_last_hof_csv(hof_dir: Path) -> Path | None:
    """Return the CSV path for the highest-numbered iteration in hof_dir."""
    if not hof_dir.exists():
        return None
    candidates = list(hof_dir.glob("hall_of_fame_iteration_*.csv"))
    if not candidates:
        return None
    # Extract integer after "iteration_"
    def iter_num(p: Path):
        m = re.search(r"hall_of_fame_iteration_(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else -1
    return max(candidates, key=iter_num)

def pick_best_row(eqs_df: pd.DataFrame) -> pd.Series | None:
    """
    Choose the row with the minimum *non-zero* 'score'.
    Fallbacks:
      1) minimum non-zero 'loss'
      2) first row with a non-empty 'equation'
    """
    if eqs_df.empty:
        return None

    EPS = 1e-12  # treat |value| <= EPS as zero (numerical noise)

    def best_index_for(col: str):
        if col not in eqs_df.columns:
            return None
        vals = pd.to_numeric(eqs_df[col], errors="coerce")
        # finite & nonzero by EPS
        mask = vals.notna() & np.isfinite(vals) & (np.abs(vals) > EPS)
        if mask.any():
            return vals[mask].idxmin()
        return None

    # 1) Prefer lowest nonzero score
    idx = best_index_for("score")
    if idx is not None:
        return eqs_df.loc[idx]

    # 2) Otherwise lowest nonzero loss
    idx = best_index_for("loss")
    if idx is not None:
        return eqs_df.loc[idx]

    # 3) Last resort: first row with equation present
    if "equation" in eqs_df.columns:
        first_valid = eqs_df["equation"].first_valid_index()
        if first_valid is not None:
            return eqs_df.loc[first_valid]

    # Nothing usable
    return None


def make_sympy_predictor(eqn_str: str, var_names: list[str]):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor
    import numpy as np

    symbols = sp.symbols(" ".join(var_names), real=True)
    local_dict = {name: sym for name, sym in zip(var_names, symbols)}
    local_dict.update({
        'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log, 'abs': sp.Abs,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'tanh': sp.tanh, 'cosh': sp.cosh, 'sinh': sp.sinh,
        'sign': sp.sign,
        # simple ReLU; adjust if your runs used a different definition:
        'relu': lambda x: sp.Piecewise((0, x < 0), (x, True)),
        'pow': sp.Pow,
        # Add any other functions you allowed in PySR here
    })

    expr = parse_expr(
        str(eqn_str),
        local_dict=local_dict,
        transformations=standard_transformations + (convert_xor,)
    )
    f = sp.lambdify(symbols, expr, modules='numpy')

    def predictor(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        cols = [X[:, i] for i in range(X.shape[1])]
        y = f(*cols)

        # Normalize shapes:
        # - scalars or length-1 arrays -> broadcast to n
        # - matrices -> ravel
        if np.isscalar(y):
            y = np.full(n, float(y), dtype=float)
        else:
            y = np.asarray(y)
            if y.ndim > 1:
                y = y.ravel()
            if y.size == 1:
                y = np.full(n, float(y.item()), dtype=float)

        # Final sanity: if still wrong length, try broadcasting
        if y.size != n:
            try:
                y = np.broadcast_to(y, (n,)).astype(float)
            except Exception:
                raise ValueError(f"Predicted shape {y.shape} cannot be matched to n={n}")
        return y.astype(float)

    return predictor

# ---------------------------
# Main
# ---------------------------

def main():
    overall_start = time.time()
    for ds in DATASETS:
        print(f"\n=== Processing dataset: {ds['prefix']} ===")
        # Load targets
        y_train = pd.read_pickle(ds['train_y'])
        y_test  = pd.read_pickle(ds['test_y'])

        # Load raw input dicts and clean columns exactly as training did
        with open(ds["train_input_dict"], "rb") as f:
            train_input_dict = pickle.load(f)
        with open(ds["test_input_dict"], "rb") as f:
            test_input_dict = pickle.load(f)

        # Clean names
        train_clean = {k: clean_column_names(df.copy()) for k, df in train_input_dict.items()}
        test_clean  = {k: clean_column_names(df.copy()) for k, df in test_input_dict.items()}

        # Paths
        hof_root = Path(f'4_Model_results/{ds["path"]}/pysr/HOF_all_iterations')
        out_dir  = Path(f'3_Data_intermediates/{ds["path"]}')
        plots_dir = Path(f'5_Plots/{ds["path"]}/pysr')  # not used here, but keeping consistency
        results_dir = Path(f'4_Model_results/{ds["path"]}/pysr')
        out_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        # Use explicit, stable order over keys
        for key in sorted(train_clean.keys()):
            print(f"  - Key: {key}")
            X_train = train_clean[key].values
            X_test  = test_clean[key].values
            var_names = train_clean[key].columns.tolist()

            hof_dir = hof_root / key
            last_csv = find_last_hof_csv(hof_dir)
            if last_csv is None:
                print(f"    ! No HOF files found in {hof_dir}. Skipping.")
                continue

            eqs_df = pd.read_csv(last_csv)
            if 'equation' not in eqs_df.columns:
                print(f"    ! 'equation' column not found in {last_csv}. Skipping.")
                continue

            best = pick_best_row(eqs_df)
            if best is None:
                print(f"    ! Empty HOF in {last_csv}. Skipping.")
                continue

            equation_str = str(best['equation'])
            score_val = best['score'] if 'score' in best.index else np.nan
            loss_val  = best['loss']  if 'loss'  in best.index else np.nan
            complexity = best['complexity'] if 'complexity' in best.index else np.nan

            # Build predictor and evaluate
            try:
                predictor = make_sympy_predictor(equation_str, var_names)
                yhat_train = predictor(X_train)
                yhat_test  = predictor(X_test)
                rmse_train = float(root_mean_squared_error(y_train, yhat_train))
                rmse_test  = float(root_mean_squared_error(y_test,  yhat_test))
            except Exception as e:
                print(f"    ! Failed to evaluate equation for {key}: {e}")
                rmse_train = np.nan
                rmse_test  = np.nan
                yhat_train = np.full_like(y_train, np.nan, dtype=float)
                yhat_test  = np.full_like(y_test,  np.nan, dtype=float)

            # Save predictions for this key
            pd.DataFrame({"yhat_train": yhat_train}).to_pickle(
                out_dir / f'{ds["prefix"]}_best_training_predictions_sympy_{key}.pkl'
            )
            pd.DataFrame({"yhat_test": yhat_test}).to_pickle(
                out_dir / f'{ds["prefix"]}_best_test_predictions_sympy_{key}.pkl'
            )

            summary_rows.append({
                "Key": key,
                "Equation": equation_str,
                "Score(min)": float(score_val) if pd.notna(score_val) else np.nan,
                "Loss": float(loss_val) if pd.notna(loss_val) else np.nan,
                "Complexity": float(complexity) if pd.notna(complexity) else np.nan,
                "Train RMSE": rmse_train,
                "Test RMSE": rmse_test,
                "HOF CSV": str(last_csv)
            })

        # Write per-dataset summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values(by=["Test RMSE"], na_position="last")
            summary_path = results_dir / f'{ds["prefix"]}_best_equation_rmse_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  -> Wrote summary: {summary_path}")
        else:
            print("  ! No keys summarized (missing HOFs or errors).")

    print(f"\nAll done in {time.time() - overall_start:.1f}s.")

if __name__ == "__main__":
    main()



