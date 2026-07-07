import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def calculate_mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))

def calculate_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))

def calculate_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)

def compile_all_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred)
    }
