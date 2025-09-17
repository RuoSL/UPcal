import os
import numpy as np
import pandas as pd
import pint
import joblib

from utils import split_data
from model_trainer import train_model, save_model
from grid_search import multi_stage_inversion
from visualizer import plot_results
from exporter import export_predictions
from sklearn.gaussian_process.kernels import RBF, Matern


# === 1. Setup ===
ureg = pint.UnitRegistry()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 2. Paths ===
train_data_path = os.path.join(BASE_DIR, "Data", "Numerical_data", "database.xlsx")
exp_data_path = os.path.join(BASE_DIR, "Data", "Experimental_data", "ExperimentalDatabase.xlsx")

# === 3. Configs ===
# You can change features, targets, param_ranges, tolerance, exp_sheet, model_type
type_configs = {
    "E": {
        "features": ['Eblock(GPa)', 'kn(Gpa/m)'],
        "targets": ['E(GPa)'],
        "param_ranges_coarse": [np.arange(15, 115.1, 1), np.arange(1e13, 1.01e15, 1e14)],
        "steps_fine": [0.1],
        "tolerance": 0.0001,
        "exp_sheet": "Igneous rocks",
        "model_type": "GPR"
    },
    "μ": {
        "features": ['μblock', 'kn/ks'],
        "targets": ['μ'],
        "param_ranges_coarse": [np.arange(0.08, 0.331, 0.01), np.arange(1, 4.1, 0.1)],
        "steps_fine": [0.05],
        "tolerance": 0.0001,
        "exp_sheet": "Sedimentary rocks",
        "model_type": "GPR"
    },
    "Strength": {
        "features": ['cc(MPa)', 'cφ(°)', 'ct(MPa)'],
        "targets": ['UCS(MPa)', 'BTS(MPa)', 'c(MPa)', 'φ(°)'],
        "param_ranges_coarse": [
            np.arange(2.5, 65.1, 1),
            np.arange(5, 50.1, 1),
            np.arange(6, 66.1, 1)
        ],
        "steps_fine": [0.1, 0.01],
        "tolerance": 0.001,
        "exp_sheet": "Metamorphic rocks",
        "model_type": "GPR"
    }
}

# === 4. Load training data ===
# You can change TRAIN_SHEET if needed: "4mm", "3mm", etc.
TRAIN_SHEET = "4mm"
train_df_all = pd.read_excel(train_data_path, sheet_name=TRAIN_SHEET)
train_df_all["Type"] = train_df_all["Type"].astype(str).str.strip()

print(f"✔ Loaded training data: {train_data_path} | Sheet: {TRAIN_SHEET}")
print(train_df_all["Type"].value_counts())

# === 5. Loop over selected types ===
selected_types = ["E", "μ", "Strength"]

for t in selected_types:
    print(f"\n=== Running block: {t} ===")
    config = type_configs[t]

    features = config["features"]
    targets = config["targets"]
    param_ranges_coarse = config["param_ranges_coarse"]
    steps_fine = config["steps_fine"]
    tolerance = config["tolerance"]
    exp_sheet = config["exp_sheet"]
    model_type = config["model_type"]

    multi_output = len(targets) > 1

    # === Set model-specific params ===
    if model_type == "RF":
        model_params = {"n_estimators": 300, "random_state": 0}
        search_space = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30]
        }

    elif model_type == "GPR":
        model_params = {
            "kernel": 1.0 * RBF(length_scale=1.0),
            "n_restarts_optimizer": 10,
            "alpha": 1e-2
        }
        search_space = {
            "kernel": [
                RBF(length_scale=1.0),
                Matern(length_scale=1.0, nu=1.5),
            ],
            "alpha": [1e-10, 1e-5, 1e-2],
            "n_restarts_optimizer": [5, 10, 20]
        }

    elif model_type == "SVR":
        model_params = {"C": 1, "epsilon": 0.1, "kernel": "rbf"}
        search_space = {
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2],
            "kernel": ["linear", "rbf", "poly"]
        }

    elif model_type == "DNN":
        model_params = {
            "hidden_layer_sizes": (64, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1000,
            "learning_rate_init": (0.001,0.01),
            "random_state": 42
        }
        search_space = {
            "hidden_layer_sizes": [(64, 64), (128, 128)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001]
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    exp_df_all = pd.read_excel(exp_data_path, sheet_name=exp_sheet)
    print(f"[INFO] Using experimental sheet: {exp_sheet}")

    df = train_df_all[train_df_all["Type"] == t].dropna()
    X_train, X_test, y_train, y_test = split_data(df, features, targets)

    model, y_pred, mse, r2, scaler_X, scaler_Y = train_model(
        model_type, model_params,
        X_train, y_train, X_test, y_test,
        use_random_search=True,
        search_space=search_space,
        n_iter_search=5,
        multi_output=multi_output
    )
    print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}")

    RESULTS_DIR = os.path.join(BASE_DIR, "Results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_name_prefix = f"{t}_{model_type}"

    save_model(model, os.path.join(RESULTS_DIR, f"{result_name_prefix}_model.pkl"))
    joblib.dump(scaler_X, os.path.join(RESULTS_DIR, f"{result_name_prefix}_scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(RESULTS_DIR, f"{result_name_prefix}_scaler_Y.pkl"))

    plot_results(y_test, y_pred, f"{result_name_prefix} Test Set")
    export_predictions(y_test, y_pred,
                       os.path.join(RESULTS_DIR, f"{result_name_prefix}_predictions.xlsx"))

    # === Inversion ===
    target_df = exp_df_all[targets]
    inversion_results = multi_stage_inversion(
        model, scaler_X, scaler_Y,
        target_df,
        param_ranges_coarse,
        steps_fine,
        tolerance
    )

    inversion_df = pd.DataFrame(inversion_results)
    inversion_df.to_excel(
        os.path.join(RESULTS_DIR, f"{result_name_prefix}_inversion.xlsx"),
        index=False
    )
    print(f"[INFO] Inversion results saved for {result_name_prefix}")

print("\nAll done!")
