from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

def build_model(model_type, model_params, multi_output=False):
    """Initialize and return a regression model based on the given model_type."""
    if model_type == "DNN":
        return MLPRegressor(**model_params)
    elif model_type == "RF":
        return RandomForestRegressor(**model_params)
    elif model_type == "SVR":
        base_model = SVR(**model_params)
        if multi_output:
            return MultiOutputRegressor(base_model)
        else:
            return base_model
    elif model_type == "GPR":
        return GaussianProcessRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(model_type, model_params, X_train, y_train, X_test, y_test,
                use_random_search=False, search_space=None, n_iter_search=5, multi_output=False):

    # === 1. 预处理：标准化 ===
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    if len(y_train.shape) == 1:
        y_train_scaled = scaler_Y.fit_transform(y_train.reshape(-1, 1)).ravel()
    elif y_train.shape[1] == 1:
        y_train_scaled = scaler_Y.fit_transform(y_train).ravel()
    else:
        y_train_scaled = scaler_Y.fit_transform(y_train)

    # === 2. 构建基础模型 ===
    model = build_model(model_type, model_params, multi_output=multi_output)

    # === 3. 超参数优化 ===
    if search_space:
        if use_random_search:
            searcher = RandomizedSearchCV(
                estimator=model,
                param_distributions=search_space,
                n_iter=n_iter_search,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                random_state=42
            )
        else:
            searcher = GridSearchCV(
                estimator=model,
                param_grid=search_space,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )
        searcher.fit(X_train_scaled, y_train_scaled)
        print("[INFO] Best Params:", searcher.best_params_)
        model = searcher.best_estimator_
    else:
        model.fit(X_train_scaled, y_train_scaled)

    # === 4. 预测与反归一化 ===
    y_pred_scaled = model.predict(X_test_scaled)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred = scaler_Y.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mse, r2, scaler_X, scaler_Y

def save_model(model, path):
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")
