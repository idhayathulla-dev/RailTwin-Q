import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ai.delay_prediction.utils import train_logger

class HyperparameterTuner:
    @staticmethod
    def tune_lightgbm(X, y) -> dict:
        """
        Tunes LightGBM regressor using RandomizedSearchCV on training samples.
        """
        train_logger.info("Starting hyperparameter tuning for LightGBM...")
        
        param_dist = {
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "max_depth": [4, 6, 8, 10],
            "num_leaves": [15, 31, 63, 127],
            "n_estimators": [50, 100, 150],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_samples": [20, 50, 100]
        }

        # Select a sub-sample of data for faster tuning if dataset is large
        if len(X) > 10000:
            indices = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y

        lgb = LGBMRegressor(verbosity=-1, random_state=42)
        search = RandomizedSearchCV(
            estimator=lgb,
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_sample, y_sample)
        train_logger.info(f"LightGBM best MAE: {-search.best_score_:.4f}")
        train_logger.info(f"LightGBM best params: {search.best_params_}")
        return search.best_params_

    @staticmethod
    def tune_xgboost(X, y) -> dict:
        """
        Tunes XGBoost regressor using RandomizedSearchCV.
        """
        train_logger.info("Starting hyperparameter tuning for XGBoost...")
        
        param_dist = {
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "max_depth": [4, 6, 8, 10],
            "n_estimators": [50, 100, 150],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0]
        }

        if len(X) > 10000:
            indices = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y

        xgb = XGBRegressor(random_state=42, verbosity=0)
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_sample, y_sample)
        train_logger.info(f"XGBoost best MAE: {-search.best_score_:.4f}")
        train_logger.info(f"XGBoost best params: {search.best_params_}")
        return search.best_params_
