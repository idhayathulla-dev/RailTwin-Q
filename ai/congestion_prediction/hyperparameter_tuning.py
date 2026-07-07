import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ai.congestion_prediction.utils import train_logger

class HyperparameterTuner:
    @staticmethod
    def tune_model(model_name: str, X, y) -> dict:
        """
        Tunes hyperparameters for the requested regressor using RandomizedSearchCV.
        """
        train_logger.info(f"Tuning hyperparameters for {model_name}...")
        
        # Sub-sample if dataset is too large
        if len(X) > 10000:
            indices = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y

        if model_name == "LightGBM":
            estimator = LGBMRegressor(verbosity=-1, random_state=42)
            param_dist = {
                "learning_rate": [0.01, 0.05, 0.1, 0.15],
                "max_depth": [4, 6, 8, 10],
                "num_leaves": [15, 31, 63, 127],
                "n_estimators": [50, 100, 150],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0]
            }
        elif model_name == "XGBoost":
            estimator = XGBRegressor(random_state=42, verbosity=0)
            param_dist = {
                "learning_rate": [0.01, 0.05, 0.1, 0.15],
                "max_depth": [4, 6, 8, 10],
                "n_estimators": [50, 100, 150],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0]
            }
        elif model_name == "RandomForest":
            estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_dist = {
                "n_estimators": [30, 50, 100],
                "max_depth": [4, 6, 8, 10],
                "min_samples_split": [2, 5, 10]
            }
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_sample, y_sample)
        train_logger.info(f"{model_name} best parameters: {search.best_params_} | Best MAE: {-search.best_score_:.4f}")
        return search.best_params_
